"""
Frontend dashboard for multi-variant ablation inference and analysis.
Backend inference, preprocessing, transforms, and metrics logic remain unchanged.
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_registry import create_model
from utils.dataset_loader import BrainMRIDataset
from models.metrics import compute_segmentation_metrics
from utils.visualization import mri_window_to_uint8

st.set_page_config(page_title="Brain MRI Tumor Segmentation", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
    .main { background: #f6f8fb; }
    [data-testid="stSidebar"] { background: #f2f5f9; border-right: 1px solid #dde4ec; }
    h1, h2, h3 { color: #1c2a39; letter-spacing: .2px; }
    .panel {
        background: white; border: 1px solid #dfe6ee; border-radius: 12px;
        padding: 12px 14px; box-shadow: 0 1px 3px rgba(19,35,61,.08);
    }
    .small-note { color: #5a6b7f; font-size: 12px; }
    .section-divider { margin: 8px 0 8px 0; border-top: 1px solid #e5ebf2; }
</style>
""",
    unsafe_allow_html=True,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path(os.environ.get("BRAIN_MRI_DATA_DIR", "E:/Brain_MRI_DL/processed_data"))
METADATA_CSV = DATA_DIR / "metadata.csv"

CLASS_NAMES = ["Background", "Necrotic Core", "Edema", "Enhancing Tumor"]
MODALITY_LABELS = ["T1", "T1CE", "T2", "FLAIR"]
CLASS_COLORS = {0: (185, 188, 192), 1: (214, 72, 86), 2: (82, 178, 126), 3: (74, 132, 218)}

# Checkpoint/model map. Add future ablation variants here only.
MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "full": {
        "label": "Full (Ours)",
        "model_key": "ablation_full",
        "checkpoint": "outputs/ablation/full/best_model.pth",
        "description": "MobileNetV2 encoder + Lightweight ASPP + Attention gates",
    },
    "no_attention": {
        "label": "No Attention",
        "model_key": "ablation_no_attention",
        "checkpoint": "outputs/ablation/no_attention/best_model.pth",
        "description": "MobileNetV2 encoder + Lightweight ASPP + direct skips",
    },
    "no_aspp": {
        "label": "No ASPP",
        "model_key": "ablation_no_aspp",
        "checkpoint": "outputs/ablation/no_aspp/best_model.pth",
        "description": "MobileNetV2 encoder + 1x1 bottleneck + Attention gates",
    },
}


def create_colored_mask(mask: np.ndarray, class_colors: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cid, (r, g, b) in class_colors.items():
        colored[mask == cid] = [r, g, b]
    return colored


def create_overlay(image_gray: np.ndarray, mask_colored: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    base = np.stack([image_gray.astype(np.float32)] * 3, axis=-1) / 255.0
    mask = mask_colored.astype(np.float32) / 255.0
    out = base * (1 - alpha) + mask * alpha
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def run_inference(model, image_tensor):
    # Keep current inference logic exactly as stabilized.
    with torch.no_grad():
        output = model(image_tensor.to(DEVICE))
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1)
    return pred.squeeze().cpu().numpy(), probs.squeeze().cpu().numpy()


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict:
    preds_t = torch.from_numpy(predictions).unsqueeze(0)
    targets_t = torch.from_numpy(targets).unsqueeze(0)
    return compute_segmentation_metrics(preds_t, targets_t, num_classes=4)


def resolve_checkpoint_path(checkpoint_rel: str) -> Optional[Path]:
    # Exact path first (expected structure).
    direct = Path(checkpoint_rel)
    if direct.exists():
        return direct
    # Fallback for accidental nested runs (older experiments).
    variant = Path(checkpoint_rel).parent.name
    base = Path("outputs") / "ablation" / variant
    if base.exists():
        cands = sorted(base.glob("**/best_model.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            return cands[0]
    return None


@st.cache_data
def load_ablation_table() -> pd.DataFrame:
    csv_path = Path("results/ablation/complete_ablation_results.csv")
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


@st.cache_data
def load_variant_summary(variant: str) -> Dict:
    path = Path("outputs/ablation") / variant / "summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _torch_load_compat(path: Path):
    # Handles checkpoints saved in environments that reference numpy._core.
    import numpy as _np

    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = _np.core
    if "numpy._core.multiarray" not in sys.modules and hasattr(_np.core, "multiarray"):
        sys.modules["numpy._core.multiarray"] = _np.core.multiarray
    return torch.load(path, map_location=DEVICE, weights_only=False)


@st.cache_resource
def load_model(model_key: str, checkpoint_rel: str):
    ckpt = resolve_checkpoint_path(checkpoint_rel)
    if ckpt is None:
        return None, f"Checkpoint not found: {checkpoint_rel}"
    try:
        model = create_model(model_key=model_key, in_channels=4, num_classes=4, pretrained=False)
        checkpoint = _torch_load_compat(ckpt)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model = model.to(DEVICE).eval()
        return model, None
    except Exception as e:
        return None, f"Error loading {ckpt}: {e}"


@st.cache_resource
def load_dataset():
    if not DATA_DIR.exists():
        st.error(f"Data directory not found: {DATA_DIR}")
        st.stop()
    if not METADATA_CSV.exists():
        st.error(f"Metadata CSV not found: {METADATA_CSV}")
        st.stop()
    return BrainMRIDataset(
        data_dir=str(DATA_DIR),
        metadata_csv=str(METADATA_CSV),
        split="val",
        val_split=0.2,
        test_split=0.1,
        seed=42,
    )


def list_pngs(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob("*.png"))


def metric_from_summary(summary: Dict, key: str, default="N/A") -> str:
    v = summary.get(key, None)
    if v is None:
        return default
    try:
        return f"{float(v):.4f}" if key.startswith("best_") else str(v)
    except Exception:
        return str(v)


# Header
st.markdown(
    """
<div style="text-align:center; padding-top:4px; padding-bottom:8px;">
  <h1 style="margin-bottom:4px;">Brain MRI Tumor Segmentation</h1>
  <p class="small-note">Multi-modal medical image segmentation dashboard for controlled ablation analysis</p>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Sidebar sections (exact requested order)
with st.sidebar:
    # 1. Model Selection
    st.markdown("### Model Selection")
    selected_variant = st.selectbox(
        "Variant",
        list(MODEL_CONFIGS.keys()),
        format_func=lambda k: MODEL_CONFIGS[k]["label"],
    )
    cfg = MODEL_CONFIGS[selected_variant]
    ckpt_resolved = resolve_checkpoint_path(cfg["checkpoint"])
    summary = load_variant_summary(selected_variant)
    if ckpt_resolved is None:
        st.warning("Checkpoint missing")
    else:
        st.success("Checkpoint available")
    st.caption(f"Model key: `{cfg['model_key']}`")
    st.caption(f"Checkpoint: `{ckpt_resolved if ckpt_resolved else cfg['checkpoint']}`")
    st.caption(cfg["description"])
    c1 = st.columns(1)[0]
    c1.metric("Params", f"{summary.get('total_params', 'N/A'):,}" if summary.get("total_params") else "N/A")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # 2. MRI Modality Display
    st.markdown("### MRI Modality Display")
    modality_idx = MODALITY_LABELS.index(
        st.radio("Modality", MODALITY_LABELS, horizontal=True, label_visibility="collapsed")
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # 3. Display Controls
    st.markdown("### Display Controls")
    show_input = st.checkbox("Show Input MRI", True)
    show_gt = st.checkbox("Show Ground Truth", True)
    show_pred = st.checkbox("Show Prediction", True)
    show_overlay = st.checkbox("Show Overlay", True)
    overlay_alpha = st.slider(
        "Overlay transparency",
        0.0,
        1.0,
        0.55,
        0.05,
        help="Only affects the Overlay panel: 0 = pure MRI, 1 = pure mask colors.",
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # 4. Window / Contrast Controls
    st.markdown("### Window / Contrast Controls")
    win_low = st.slider(
        "Low percentile",
        0.0,
        50.0,
        2.0,
        0.5,
        help="Pixels below this percentile are clipped to black (contrast cleanup).",
    )
    win_high = st.slider(
        "High percentile",
        50.0,
        100.0,
        98.0,
        0.5,
        help="Pixels above this percentile are clipped to white (prevents washout).",
    )
    if st.button("Refresh inference", use_container_width=True):
        st.session_state.force_infer = True


tab1, tab2, tab3 = st.tabs(["Interactive Segmentation", "Metrics & Analysis", "About"])

dataset = load_dataset()
model, load_warning = load_model(cfg["model_key"], cfg["checkpoint"])

with tab1:
    top_l, top_r = st.columns([1, 2.4], gap="medium")
    with top_l:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Sample Selection")
        st.success(f"Dataset loaded: {len(dataset)} samples")
        sample_idx = st.slider("Sample index", 0, len(dataset) - 1, 0, 1)
        st.markdown("</div>", unsafe_allow_html=True)
        if load_warning:
            st.warning(load_warning)
    with top_r:
        image, mask = dataset[sample_idx]
        win_hi = max(win_high, win_low + 2.0)
        image_display = mri_window_to_uint8(image[modality_idx].numpy(), p_low=win_low, p_high=win_hi)

        prev_idx = st.session_state.setdefault("infer_sample_idx", None)
        prev_variant = st.session_state.setdefault("infer_model_variant", None)
        force = st.session_state.pop("force_infer", False)
        need_infer = force or prev_idx != sample_idx or prev_variant != selected_variant

        if model is not None and need_infer:
            with st.spinner("Running inference"):
                pred_mask, _ = run_inference(model, image.unsqueeze(0))
                st.session_state.pred_mask = pred_mask
                st.session_state.metrics = compute_metrics(pred_mask, mask.numpy())
                st.session_state.infer_sample_idx = sample_idx
                st.session_state.infer_model_variant = selected_variant

        pred_mask = st.session_state.get("pred_mask", None)
        if model is None:
            st.info("Inference is disabled until the selected checkpoint is available and loadable.")
        elif pred_mask is None:
            st.info("Click refresh inference.")
        else:
            fig, axes = plt.subplots(1, 4, figsize=(16.8, 4.2), constrained_layout=True)
            fig.patch.set_facecolor("#f6f8fb")

            if show_input:
                axes[0].imshow(image_display, cmap="gray", interpolation="nearest")
            axes[0].set_title("Input MRI", loc="center")
            axes[0].axis("off")

            if show_gt:
                gt_colored = create_colored_mask(mask.numpy(), CLASS_COLORS)
                axes[1].imshow(gt_colored, interpolation="nearest")
            axes[1].set_title("Ground Truth", loc="center")
            axes[1].axis("off")

            if show_pred:
                pred_colored = create_colored_mask(pred_mask, CLASS_COLORS)
                axes[2].imshow(pred_colored, interpolation="nearest")
            axes[2].set_title("Prediction", loc="center")
            axes[2].axis("off")

            if show_overlay:
                overlay = create_overlay(image_display, create_colored_mask(pred_mask, CLASS_COLORS), overlay_alpha)
                axes[3].imshow(overlay, interpolation="nearest")
            axes[3].set_title("Overlay", loc="center")
            axes[3].axis("off")

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Ablation Comparison Table")
    ablation_df = load_ablation_table()
    if ablation_df.empty:
        st.warning("Ablation CSV not found at results/ablation/complete_ablation_results.csv")
    else:
        ablation_show = ablation_df.rename(
            columns={
                "Variant": "Variant",
                "Total Params": "Params",
                "GFLOPs": "GFLOPs",
                "Latency (ms)": "Latency",
                "Best Dice": "Dice",
                "Best IoU": "IoU",
                "Description": "Notes",
            }
        )[["Variant", "Params", "GFLOPs", "Latency", "Dice", "IoU", "Notes"]]
        st.dataframe(ablation_show, use_container_width=True, hide_index=True)

with tab2:
    st.markdown("## Dataset Analysis")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Original Train", "57,209")
    c2.metric("Filtered Train", "20,000")
    c3.metric("Retention", "34.96%")
    c4.metric("Original Background", "97.41%")
    c5.metric("Filtered Background", "95.10%")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original Edema", "1.63%")
    c2.metric("Filtered Edema", "2.90%")
    c3.metric("Original Enhancing", "0.57%")
    c4.metric("Filtered Enhancing", "1.13%")

    st.markdown("#### Original Dataset Analysis")
    dataset_plot_dir = Path("results/dataset_analysis/plots")
    dataset_plots = list_pngs(dataset_plot_dir)
    if not dataset_plots:
        st.warning(f"No dataset analysis plots found in {dataset_plot_dir}")
    else:
        cols = st.columns(2)
        for i, p in enumerate(dataset_plots):
            with cols[i % 2]:
                st.image(str(p), caption=p.name, use_column_width=True)

    st.markdown("#### Filtered Dataset Analysis")
    # Prefer workspace-local folder provided by user, then fallback to dataset drive path.
    filtered_plot_candidates = [
        Path("old_dataset_graphs/tumor_density_original_vs_filtered.png"),
        Path("old_dataset_graphs/retention_by_density_bucket.png"),
        Path("old_dataset_graphs/class_distribution_original_vs_filtered.png"),
        Path("E:/Brain_MRI_DL/processed_data_filtered_v1/plots/tumor_density_original_vs_filtered.png"),
        Path("E:/Brain_MRI_DL/processed_data_filtered_v1/plots/retention_by_density_bucket.png"),
        Path("E:/Brain_MRI_DL/processed_data_filtered_v1/plots/class_distribution_original_vs_filtered.png"),
    ]
    existing_filtered = [p for p in filtered_plot_candidates if p.exists()]
    if existing_filtered:
        cols = st.columns(2)
        for i, p in enumerate(existing_filtered):
            with cols[i % 2]:
                st.image(str(p), caption=p.name, use_column_width=True)
    else:
        st.info("Filtered comparison plots not found in `old_dataset_graphs` or `processed_data_filtered_v1/plots`.")

    st.markdown("#### Original vs Filtered Comparison Summary")
    st.markdown(
        """
- High tumor-density slices were retained first.
- All slices above 5% tumor coverage were retained.
- 2-5% tumor-density slices were prioritized.
- Lower-information/background-heavy slices were aggressively reduced.
- Necrotic and Enhancing class diversity retention was enforced.
"""
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## Model / Ablation Analysis")
    ablation_plot_dir = Path("results/ablation")
    ablation_plots = list_pngs(ablation_plot_dir)
    if not ablation_plots:
        st.warning(f"No ablation plots found in {ablation_plot_dir}")
    else:
        cols = st.columns(2)
        for i, p in enumerate(ablation_plots):
            with cols[i % 2]:
                st.image(str(p), caption=p.name, use_column_width=True)

with tab3:
    st.markdown("## Project Overview")
    st.markdown(
        """
This application performs real multi-class brain MRI segmentation on validation slices using trained ablation variants.

Variant definitions:
- **Full (Ours):** MobileNetV2 encoder + Lightweight ASPP + Attention gates.
- **No Attention:** Same core architecture with attention gates removed.
- **No ASPP:** Same core architecture with lightweight ASPP replaced by a simple bottleneck.
"""
    )
    st.markdown("## Operational Notes")
    st.markdown(
        """
- Inference uses the current stabilized pipeline (softmax + argmax).
- Preprocessing, transforms, dataset loading behavior, and metrics formulas are unchanged.
- New variants can be added by extending `MODEL_CONFIGS` in this file.
"""
    )