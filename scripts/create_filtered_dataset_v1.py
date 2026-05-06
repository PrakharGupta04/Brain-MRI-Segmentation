"""
Create an information-dense reduced training dataset (v1) from processed_data.

Design goals:
- Never modify original dataset.
- Keep val/test untouched (fully retained).
- Filter only train split with deterministic, tumor-density-aware strategy.
- Preserve class diversity and produce reproducible reports.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass(frozen=True)
class DatasetLayout:
    data_dir: Path
    metadata_csv: Path
    images_dir: Path
    masks_dir: Path


CLASS_NAMES = {0: "Background", 1: "Necrotic", 2: "Edema", 3: "Enhancing"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create filtered train split while preserving full val/test."
    )
    parser.add_argument(
        "--source_dir",
        default=r"E:\Brain_MRI_DL\processed_data",
        help="Original processed dataset root (read-only).",
    )
    parser.add_argument(
        "--output_dir",
        default=r"E:\Brain_MRI_DL\processed_data_filtered_v1",
        help="Standalone destination dataset root.",
    )
    parser.add_argument(
        "--target_train_size",
        type=int,
        default=20000,
        help="Approximate retained train slice count.",
    )
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow_existing_output",
        action="store_true",
        default=False,
        help="Allow writing into existing output_dir if empty structure exists.",
    )
    return parser.parse_args()


def inspect_layout(source_dir: Path) -> DatasetLayout:
    metadata_csv = source_dir / "metadata.csv"
    images_dir = source_dir / "images"
    masks_dir = source_dir / "masks"

    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata.csv not found at {metadata_csv}")
    if not images_dir.exists():
        raise FileNotFoundError(f"images directory not found at {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"masks directory not found at {masks_dir}")

    return DatasetLayout(
        data_dir=source_dir,
        metadata_csv=metadata_csv,
        images_dir=images_dir,
        masks_dir=masks_dir,
    )


def split_patients(metadata: pd.DataFrame, val_split: float, test_split: float, seed: int) -> pd.Series:
    unique_patients = metadata["patient_id"].unique()
    rng = np.random.default_rng(seed)
    shuffled = unique_patients.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    test_n = int(n * test_split)
    val_n = int(n * val_split)
    train_n = n - test_n - val_n

    train_patients = set(shuffled[:train_n])
    val_patients = set(shuffled[train_n : train_n + val_n])

    split = []
    for pid in metadata["patient_id"]:
        if pid in train_patients:
            split.append("train")
        elif pid in val_patients:
            split.append("val")
        else:
            split.append("test")
    return pd.Series(split, index=metadata.index, name="split")


def compute_slice_stats(mask_path: Path) -> Dict:
    mask = np.load(mask_path).astype(np.int64)
    total = int(mask.size)
    counts = np.bincount(mask.reshape(-1), minlength=4)

    tumor_pixels = int(counts[1] + counts[2] + counts[3])
    tumor_pct = float(tumor_pixels / max(total, 1) * 100.0)

    return {
        "total_pixels": total,
        "tumor_pixels": tumor_pixels,
        "tumor_pct": tumor_pct,
        "background_pixels": int(counts[0]),
        "necrotic_pixels": int(counts[1]),
        "edema_pixels": int(counts[2]),
        "enhancing_pixels": int(counts[3]),
        "has_necrotic": int(counts[1] > 0),
        "has_edema": int(counts[2] > 0),
        "has_enhancing": int(counts[3] > 0),
    }


def assign_density_bucket(tumor_pct: float) -> str:
    if tumor_pct > 5.0:
        return "gt_5"
    if tumor_pct > 2.0:
        return "2_to_5"
    if tumor_pct > 1.0:
        return "1_to_2"
    if tumor_pct > 0.5:
        return "0_5_to_1"
    return "lt_0_5"


def select_train_rows(train_df: pd.DataFrame, target_size: int, seed: int) -> pd.DataFrame:
    """
    Tiered, deterministic selection:
      - Keep all >5% and 2-5% (if target permits).
      - Weighted allocation for lower tiers.
      - Class-diversity constraints for Necrotic/Enhancing.
      - Exact-size adjustment with quality-prioritized trimming/filling.
    """
    rng = np.random.default_rng(seed)
    df = train_df.copy()
    df["density_bucket"] = df["tumor_pct"].map(assign_density_bucket)
    df["quality_score"] = df["tumor_pct"]

    # Deterministic shuffle tie-break
    df["rand"] = rng.random(len(df))
    df = df.sort_values(["quality_score", "rand"], ascending=[False, True]).reset_index(drop=True)

    high = df[df["density_bucket"].isin(["gt_5", "2_to_5"])].copy()
    low = df[~df["density_bucket"].isin(["gt_5", "2_to_5"])].copy()

    if len(high) >= target_size:
        selected = high.head(target_size).copy()
    else:
        selected_parts = [high]
        remaining = target_size - len(high)

        bucket_weights = {
            "1_to_2": 0.80,
            "0_5_to_1": 0.50,
            "lt_0_5": 0.25,
        }
        alloc_candidates = []
        for b, w in bucket_weights.items():
            sub = low[low["density_bucket"] == b].copy()
            alloc_candidates.append((b, sub, w))

        denom = sum(len(sub) * w for _, sub, w in alloc_candidates)
        kept_indices = set()
        for _, sub, w in alloc_candidates:
            if len(sub) == 0:
                continue
            if denom <= 0:
                n_take = 0
            else:
                n_take = int(round(remaining * (len(sub) * w) / denom))
            n_take = min(n_take, len(sub))
            chosen = sub.head(n_take)
            kept_indices.update(chosen.index.tolist())

        selected_low = low.loc[sorted(kept_indices)].copy()

        if len(selected_low) < remaining:
            fill_pool = low.drop(index=selected_low.index)
            selected_low = pd.concat([selected_low, fill_pool.head(remaining - len(selected_low))], axis=0)

        selected = pd.concat([high, selected_low], axis=0)
        if len(selected) > target_size:
            selected = selected.sort_values(["quality_score", "rand"], ascending=[False, True]).head(target_size)

    # Class diversity enforcement (focus on minority classes).
    selected = selected.copy()
    global_keep_ratio = min(1.0, target_size / max(len(df), 1))
    min_ratio = max(0.25, global_keep_ratio * 0.95)

    for class_col in ["has_necrotic", "has_enhancing"]:
        total_with_class = int(df[class_col].sum())
        required = int(round(total_with_class * min_ratio))
        current = int(selected[class_col].sum())
        if current >= required:
            continue

        need = required - current
        add_pool = df[(df[class_col] == 1) & (~df.index.isin(selected.index))].sort_values(
            ["quality_score", "rand"], ascending=[False, True]
        )
        add_rows = add_pool.head(need)
        selected = pd.concat([selected, add_rows], axis=0)

        # Trim back to target while protecting required class counts.
        while len(selected) > target_size:
            removable = selected[
                (selected[class_col] == 0)
                | (selected[class_col] == 1)
                & (selected[class_col].sum() > required)
            ]
            if len(removable) == 0:
                break
            worst_idx = removable.sort_values(["quality_score", "rand"], ascending=[True, False]).index[0]
            selected = selected.drop(index=worst_idx)

    # Final exact sizing.
    selected = selected.sort_values(["quality_score", "rand"], ascending=[False, True])
    if len(selected) > target_size:
        selected = selected.head(target_size)
    elif len(selected) < target_size:
        extra = df[~df.index.isin(selected.index)].head(target_size - len(selected))
        selected = pd.concat([selected, extra], axis=0)

    return selected.copy()


def ensure_output_structure(output_dir: Path, allow_existing: bool):
    if output_dir.exists():
        existing_files = list(output_dir.glob("**/*"))
        if existing_files and not allow_existing:
            raise FileExistsError(
                f"Output directory {output_dir} already contains files. "
                f"Use a new path or pass --allow_existing_output."
            )
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)


def copy_selected_files(df: pd.DataFrame, src_images: Path, src_masks: Path, dst_images: Path, dst_masks: Path):
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying selected files", ncols=100):
        src_img = src_images / row["image_name"]
        src_msk = src_masks / row["mask_name"]
        dst_img = dst_images / row["image_name"]
        dst_msk = dst_masks / row["mask_name"]
        if not src_img.exists() or not src_msk.exists():
            raise FileNotFoundError(f"Missing source pair: {src_img} | {src_msk}")
        if dst_img.exists() or dst_msk.exists():
            continue
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_msk, dst_msk)


def plot_comparisons(train_full: pd.DataFrame, train_sel: pd.DataFrame, out_plots: Path):
    # Tumor % histogram: original vs filtered
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(train_full["tumor_pct"], bins=60, alpha=0.45, label="Original train", color="#4575b4")
    ax.hist(train_sel["tumor_pct"], bins=60, alpha=0.60, label="Filtered train", color="#d73027")
    ax.set_title("Tumor Density Distribution: Original vs Filtered Train")
    ax.set_xlabel("Tumor coverage per slice (%)")
    ax.set_ylabel("Slice count")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_plots / "tumor_density_original_vs_filtered.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Class pixel distribution (train only)
    def class_pct(df: pd.DataFrame) -> np.ndarray:
        vals = np.array(
            [
                df["background_pixels"].sum(),
                df["necrotic_pixels"].sum(),
                df["edema_pixels"].sum(),
                df["enhancing_pixels"].sum(),
            ],
            dtype=np.float64,
        )
        return vals / max(vals.sum(), 1.0) * 100.0

    full_pct = class_pct(train_full)
    sel_pct = class_pct(train_sel)
    x = np.arange(4)
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, full_pct, width=w, label="Original train")
    ax.bar(x + w / 2, sel_pct, width=w, label="Filtered train")
    ax.set_xticks(x)
    ax.set_xticklabels(["Background", "Necrotic", "Edema", "Enhancing"])
    ax.set_ylabel("Pixel percentage (%)")
    ax.set_title("Class Pixel Distribution: Original vs Filtered Train")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_plots / "class_distribution_original_vs_filtered.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Retention by tumor-density bucket
    full_bucket = train_full["density_bucket"].value_counts().sort_index()
    sel_bucket = train_sel["density_bucket"].value_counts().sort_index()
    buckets = sorted(set(full_bucket.index.tolist()) | set(sel_bucket.index.tolist()))
    full_counts = np.array([int(full_bucket.get(b, 0)) for b in buckets], dtype=np.float64)
    sel_counts = np.array([int(sel_bucket.get(b, 0)) for b in buckets], dtype=np.float64)
    retention = np.divide(sel_counts, np.maximum(full_counts, 1.0)) * 100.0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(buckets, retention, color="#1b9e77")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Retention (%)")
    ax.set_title("Retention by Tumor-Density Bucket (Train)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_plots / "retention_by_density_bucket.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    layout = inspect_layout(source_dir)
    ensure_output_structure(output_dir, allow_existing=args.allow_existing_output)

    metadata = pd.read_csv(layout.metadata_csv)
    required_cols = {"image_name", "mask_name", "patient_id"}
    missing = required_cols - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata.csv missing required columns: {sorted(missing)}")

    metadata["split"] = split_patients(metadata, args.val_split, args.test_split, args.seed)

    # Analyze train slice-level tumor density / class presence.
    train_df = metadata[metadata["split"] == "train"].copy().reset_index(drop=True)
    stats_rows: List[Dict] = []
    for i, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Analyzing train masks", ncols=100):
        stat = compute_slice_stats(layout.masks_dir / row["mask_name"])
        stat["row_id"] = int(i)
        stats_rows.append(stat)
    stats_df = pd.DataFrame(stats_rows)
    train_df = pd.concat([train_df, stats_df], axis=1)
    train_df["density_bucket"] = train_df["tumor_pct"].map(assign_density_bucket)

    # Select filtered train set.
    selected_train = select_train_rows(train_df, args.target_train_size, args.seed)
    selected_train = selected_train.sort_values("row_id").reset_index(drop=True)

    # Keep val/test untouched (all rows retained).
    val_test_df = metadata[metadata["split"].isin(["val", "test"])].copy()

    final_metadata = pd.concat(
        [
            selected_train[metadata.columns.tolist()],
            val_test_df[metadata.columns.tolist()],
        ],
        axis=0,
    ).reset_index(drop=True)

    # Copy selected files (standalone dataset).
    copy_selected_files(
        final_metadata,
        src_images=layout.images_dir,
        src_masks=layout.masks_dir,
        dst_images=output_dir / "images",
        dst_masks=output_dir / "masks",
    )

    # Write metadata expected by pipeline (same schema) + split-aware artifact.
    final_metadata[metadata.columns].to_csv(output_dir / "metadata.csv", index=False)
    final_metadata.to_csv(output_dir / "reports" / "metadata_with_split.csv", index=False)
    selected_train.to_csv(output_dir / "reports" / "retained_train_slices.csv", index=False)

    # Summary stats.
    orig_train_n = int(len(train_df))
    sel_train_n = int(len(selected_train))
    train_retention_pct = float(sel_train_n / max(orig_train_n, 1) * 100.0)

    bucket_full = train_df["density_bucket"].value_counts().to_dict()
    bucket_sel = selected_train["density_bucket"].value_counts().to_dict()
    bucket_retention = {
        b: float(bucket_sel.get(b, 0) / max(bucket_full.get(b, 1), 1) * 100.0)
        for b in sorted(set(bucket_full) | set(bucket_sel))
    }

    def cls_summary(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        pixels = {
            "background": int(df["background_pixels"].sum()),
            "necrotic": int(df["necrotic_pixels"].sum()),
            "edema": int(df["edema_pixels"].sum()),
            "enhancing": int(df["enhancing_pixels"].sum()),
        }
        tot = max(sum(pixels.values()), 1)
        pct = {k: float(v / tot * 100.0) for k, v in pixels.items()}
        return {"pixel_counts": pixels, "pixel_pct": pct}

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed": args.seed,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "target_train_size": args.target_train_size,
        "original_counts": {
            "train": orig_train_n,
            "val": int((metadata["split"] == "val").sum()),
            "test": int((metadata["split"] == "test").sum()),
            "total": int(len(metadata)),
        },
        "filtered_counts": {
            "train": sel_train_n,
            "val": int((final_metadata["split"] == "val").sum()),
            "test": int((final_metadata["split"] == "test").sum()),
            "total": int(len(final_metadata)),
        },
        "train_retention_pct": train_retention_pct,
        "density_bucket_original": bucket_full,
        "density_bucket_filtered": bucket_sel,
        "density_bucket_retention_pct": bucket_retention,
        "class_distribution_train_original": cls_summary(train_df),
        "class_distribution_train_filtered": cls_summary(selected_train),
        "class_presence_train_original": {
            "has_necrotic_slices": int(train_df["has_necrotic"].sum()),
            "has_edema_slices": int(train_df["has_edema"].sum()),
            "has_enhancing_slices": int(train_df["has_enhancing"].sum()),
        },
        "class_presence_train_filtered": {
            "has_necrotic_slices": int(selected_train["has_necrotic"].sum()),
            "has_edema_slices": int(selected_train["has_edema"].sum()),
            "has_enhancing_slices": int(selected_train["has_enhancing"].sum()),
        },
        "strategy": {
            "high_density_buckets": "gt_5 and 2_to_5 prioritized/retained first",
            "lower_density_sampling": {
                "1_to_2": 0.80,
                "0_5_to_1": 0.50,
                "lt_0_5": 0.25,
            },
            "class_diversity_enforcement": "necrotic/enhancing minimum retention relative to target keep ratio",
            "exact_target_adjustment": "quality-score based trim/fill after diversity constraints",
        },
    }
    (output_dir / "reports" / "filtering_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plots.
    plot_comparisons(train_df, selected_train, output_dir / "plots")

    print("=" * 90)
    print("FILTERED DATASET CREATED SUCCESSFULLY")
    print("=" * 90)
    print(f"Source dataset : {source_dir}")
    print(f"Output dataset : {output_dir}")
    print(f"Train slices   : {orig_train_n} -> {sel_train_n} ({train_retention_pct:.2f}% kept)")
    print(f"Val slices     : {(metadata['split'] == 'val').sum()} (fully retained)")
    print(f"Test slices    : {(metadata['split'] == 'test').sum()} (fully retained)")
    print(f"Reports        : {output_dir / 'reports'}")
    print(f"Plots          : {output_dir / 'plots'}")


if __name__ == "__main__":
    main()

