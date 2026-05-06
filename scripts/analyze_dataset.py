"""
Dataset analysis and quality profiling for controlled optimization decisions.

This script is read-only: it never modifies source data.
It profiles slice-level mask statistics for the TRAIN split only and writes:
- CSV tables
- JSON summary
- Visualization plots

Outputs:
results/dataset_analysis/
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root for local script execution (python scripts/...)
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dataset_loader import BrainMRIDataset


CLASS_NAMES = {
    0: "Background",
    1: "Necrotic",
    2: "Edema",
    3: "Enhancing",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze train-split slice statistics")
    parser.add_argument(
        "--data_dir",
        default="E:/Brain_MRI_DL/processed_data",
        help="Path to preprocessed data folder",
    )
    parser.add_argument(
        "--metadata_csv",
        default=None,
        help="Optional metadata.csv path. Defaults to <data_dir>/metadata.csv",
    )
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed")
    parser.add_argument(
        "--output_dir",
        default="results/dataset_analysis",
        help="Output analysis directory",
    )
    parser.add_argument(
        "--max_slices",
        type=int,
        default=0,
        help="Optional cap for quick analysis (0 means full train split)",
    )
    return parser.parse_args()


def dominant_class(counts: np.ndarray) -> int:
    return int(np.argmax(counts))


def compute_slice_stats(mask: np.ndarray) -> Dict:
    # mask shape: [H, W], values in {0,1,2,3}
    total_pixels = int(mask.size)
    counts = np.bincount(mask.reshape(-1), minlength=4)
    tumor_pixels = int(counts[1] + counts[2] + counts[3])
    bg_pixels = int(counts[0])

    tumor_pct = float(tumor_pixels / total_pixels * 100.0)
    bg_pct = float(bg_pixels / total_pixels * 100.0)
    class_pcts = (counts / total_pixels * 100.0).astype(np.float64)

    has_necrotic = int(counts[1] > 0)
    has_edema = int(counts[2] > 0)
    has_enhancing = int(counts[3] > 0)
    non_empty = int(tumor_pixels > 0)

    return {
        "total_pixels": total_pixels,
        "tumor_pixels": tumor_pixels,
        "tumor_pct": tumor_pct,
        "background_pixels": bg_pixels,
        "background_pct": bg_pct,
        "necrotic_pixels": int(counts[1]),
        "necrotic_pct": float(class_pcts[1]),
        "edema_pixels": int(counts[2]),
        "edema_pct": float(class_pcts[2]),
        "enhancing_pixels": int(counts[3]),
        "enhancing_pct": float(class_pcts[3]),
        "has_necrotic": has_necrotic,
        "has_edema": has_edema,
        "has_enhancing": has_enhancing,
        "non_empty_mask": non_empty,
        "is_background_only": int(tumor_pixels == 0),
        "dominant_class_id": dominant_class(counts),
        "dominant_class": CLASS_NAMES[dominant_class(counts)],
    }


def build_plots(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Histogram of tumor percentage per slice
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["tumor_pct"].values, bins=50, color="#2f7ed8", alpha=0.85, edgecolor="white")
    ax.set_title("Tumor Coverage per Slice (Train Split)")
    ax.set_xlabel("Tumor pixels (%)")
    ax.set_ylabel("Number of slices")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "tumor_pct_histogram.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 2) Class-wise global pixel distribution
    class_pixels = np.array(
        [
            df["background_pixels"].sum(),
            df["necrotic_pixels"].sum(),
            df["edema_pixels"].sum(),
            df["enhancing_pixels"].sum(),
        ],
        dtype=np.float64,
    )
    class_pct = class_pixels / np.maximum(class_pixels.sum(), 1.0) * 100.0
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        ["Background", "Necrotic", "Edema", "Enhancing"],
        class_pct,
        color=["#9aa0a6", "#d65f5f", "#58a55c", "#4f81bd"],
    )
    ax.set_title("Global Pixel Distribution by Class (Train Split)")
    ax.set_ylabel("Pixel percentage (%)")
    ax.set_ylim(0, max(100.0, float(class_pct.max() * 1.1)))
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "class_pixel_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 3) Background-only vs tumor-containing slice ratio
    bg_only = int(df["is_background_only"].sum())
    tumor_slices = int(len(df) - bg_only)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        [bg_only, tumor_slices],
        labels=["Background-only", "Tumor-containing"],
        autopct="%1.1f%%",
        colors=["#b0b0b0", "#4f81bd"],
        startangle=90,
    )
    ax.set_title("Slice Composition (Train Split)")
    fig.tight_layout()
    fig.savefig(output_dir / "background_vs_tumor_slices.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 4) Cumulative tumor coverage curve
    tumor_pixels_sorted = np.sort(df["tumor_pixels"].values)[::-1]
    total_tumor = max(tumor_pixels_sorted.sum(), 1)
    cumulative = np.cumsum(tumor_pixels_sorted) / total_tumor * 100.0
    x = np.arange(1, len(tumor_pixels_sorted) + 1) / len(tumor_pixels_sorted) * 100.0
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, cumulative, color="#1b9e77", linewidth=2.0)
    ax.set_title("Cumulative Tumor Coverage vs Slice Fraction (Train Split)")
    ax.set_xlabel("Top slices retained (%)")
    ax.set_ylabel("Cumulative tumor pixels retained (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_tumor_coverage.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def estimate_reduction_scenarios(df: pd.DataFrame) -> List[Dict]:
    total = int(len(df))
    tumor_slices = int((df["tumor_pixels"] > 0).sum())
    bg_slices = total - tumor_slices

    scenarios = []
    for keep_bg_ratio in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        keep_bg = int(round(bg_slices * keep_bg_ratio))
        kept_total = tumor_slices + keep_bg
        reduction_pct = (1.0 - (kept_total / max(total, 1))) * 100.0
        scenarios.append(
            {
                "strategy": "keep_all_tumor_plus_background_fraction",
                "keep_background_ratio": keep_bg_ratio,
                "estimated_kept_slices": kept_total,
                "estimated_removed_slices": total - kept_total,
                "estimated_reduction_pct": float(reduction_pct),
                "tumor_slice_retention_pct": 100.0,
            }
        )
    return scenarios


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    metadata_csv = Path(args.metadata_csv) if args.metadata_csv else data_dir / "metadata.csv"
    out_dir = Path(args.output_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset = BrainMRIDataset(
        data_dir=str(data_dir),
        metadata_csv=str(metadata_csv),
        split="train",
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )

    metadata = dataset.metadata.copy().reset_index(drop=True)
    if args.max_slices > 0:
        metadata = metadata.iloc[: min(args.max_slices, len(metadata))].copy()

    rows: List[Dict] = []
    for i in tqdm(range(len(metadata)), desc="Analyzing train slices", ncols=100):
        row = metadata.iloc[i]
        mask_path = dataset.masks_dir / row["mask_name"]
        mask = np.load(mask_path).astype(np.int64)
        stats = compute_slice_stats(mask)
        stats.update(
            {
                "index_in_split": int(i),
                "patient_id": str(row.get("patient_id", "")),
                "slice_idx": int(row.get("slice_idx", -1)),
                "image_name": str(row.get("image_name", "")),
                "mask_name": str(row.get("mask_name", "")),
            }
        )
        rows.append(stats)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "slice_stats_train.csv", index=False)

    # Aggregate summary
    total_slices = int(len(df))
    bg_only_slices = int(df["is_background_only"].sum())
    tumor_slices = total_slices - bg_only_slices
    total_pixels = int(df["total_pixels"].sum())
    class_pixels = {
        "background": int(df["background_pixels"].sum()),
        "necrotic": int(df["necrotic_pixels"].sum()),
        "edema": int(df["edema_pixels"].sum()),
        "enhancing": int(df["enhancing_pixels"].sum()),
    }
    class_pct = {k: float(v / max(total_pixels, 1) * 100.0) for k, v in class_pixels.items()}

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(data_dir.resolve()),
        "metadata_csv": str(metadata_csv.resolve()),
        "split_analyzed": "train",
        "val_split": args.val_split,
        "test_split": args.test_split,
        "seed": args.seed,
        "analyzed_slices": total_slices,
        "background_only_slices": bg_only_slices,
        "tumor_containing_slices": tumor_slices,
        "tumor_slice_ratio_pct": float(tumor_slices / max(total_slices, 1) * 100.0),
        "background_only_ratio_pct": float(bg_only_slices / max(total_slices, 1) * 100.0),
        "non_empty_mask_ratio_pct": float(df["non_empty_mask"].mean() * 100.0),
        "mean_tumor_pct_per_slice": float(df["tumor_pct"].mean()),
        "median_tumor_pct_per_slice": float(df["tumor_pct"].median()),
        "class_pixel_counts": class_pixels,
        "class_pixel_percentages": class_pct,
        "class_presence_counts": {
            "necrotic_present_slices": int(df["has_necrotic"].sum()),
            "edema_present_slices": int(df["has_edema"].sum()),
            "enhancing_present_slices": int(df["has_enhancing"].sum()),
        },
        "suggested_reduction_scenarios": estimate_reduction_scenarios(df),
    }

    (out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # Class distribution table for quick inspection
    class_distribution_table = pd.DataFrame(
        {
            "class": ["background", "necrotic", "edema", "enhancing"],
            "pixel_count": [
                class_pixels["background"],
                class_pixels["necrotic"],
                class_pixels["edema"],
                class_pixels["enhancing"],
            ],
            "pixel_pct": [
                class_pct["background"],
                class_pct["necrotic"],
                class_pct["edema"],
                class_pct["enhancing"],
            ],
        }
    )
    class_distribution_table.to_csv(out_dir / "class_distribution_train.csv", index=False)

    build_plots(df, plots_dir)

    print("=" * 80)
    print("DATASET ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Train slices analyzed: {total_slices}")
    print(f"Tumor-containing slices: {tumor_slices} ({summary['tumor_slice_ratio_pct']:.2f}%)")
    print(f"Background-only slices: {bg_only_slices} ({summary['background_only_ratio_pct']:.2f}%)")
    print(f"Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

