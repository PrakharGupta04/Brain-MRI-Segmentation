# ============ SCRIPTS/GENERATE_RESULTS.PY - COMPLETE CORRECTED VERSION ============

"""
Generate segmentation results on validation set samples
Saves all visualizations to results/ folder
"""

import sys
import json
import os
import argparse
import time
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_registry import (
    create_model,
    count_parameters,
    get_model_display_name,
    list_model_keys,
    load_checkpoint_weights,
)
from utils.dataset_loader import BrainMRIDataset
from utils.visualization import (
    plot_segmentation_results, 
    plot_confusion_matrix,
    plot_class_distribution,
    plot_per_class_metrics,
    create_metrics_summary_table
)
from models.metrics import SegmentationMetrics, compute_segmentation_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Generate validation segmentation results")
    parser.add_argument(
        "--model_name",
        default=os.environ.get("BRAIN_MRI_MODEL_NAME", "mobilenet_attention_unet"),
        choices=list(list_model_keys()),
        help="Architecture to evaluate",
    )
    parser.add_argument(
        "--model_path",
        default=os.environ.get("BRAIN_MRI_MODEL_PATH", "outputs/best_model.pth"),
        help="Checkpoint path",
    )
    parser.add_argument(
        "--data_dir",
        default=os.environ.get("BRAIN_MRI_DATA_DIR", "E:/Brain_MRI_DL/processed_data"),
        help="Processed dataset directory",
    )
    parser.add_argument("--num_samples", type=int, default=10, help="Number of val samples for report")
    parser.add_argument(
        "--results_root",
        default="results/experiments",
        help="Root folder for model-specific outputs",
    )
    parser.add_argument(
        "--experiment_name",
        default="latest",
        help="Subfolder inside model folder (e.g. run_01).",
    )
    return parser.parse_args()


ARGS = parse_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(ARGS.model_path)
DATA_DIR = Path(ARGS.data_dir)
METADATA_CSV = DATA_DIR / "metadata.csv"
RESULTS_DIR = Path(ARGS.results_root) / ARGS.model_name / ARGS.experiment_name
SEG_DIR = RESULTS_DIR / "segmentations"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = RESULTS_DIR / "reports"

for _d in (RESULTS_DIR, SEG_DIR, PLOTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("BRAIN MRI SEGMENTATION - RESULTS GENERATION")
print("=" * 80)
print(f"\n📍 Device: {DEVICE}")
print(f"📍 Model name: {ARGS.model_name} ({get_model_display_name(ARGS.model_name)})")
print(f"📍 Model: {MODEL_PATH}")
print(f"📍 Data: {DATA_DIR}")
print(f"📍 Results: {RESULTS_DIR}\n")


# =====================
# LOAD MODEL
# =====================
print("📂 Loading model...")
try:
    model = create_model(model_key=ARGS.model_name, in_channels=4, num_classes=4, pretrained=False)
    
    if not MODEL_PATH.exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        sys.exit(1)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    load_checkpoint_weights(model, checkpoint)
    model = model.to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully")
    total_params, trainable_params = count_parameters(model)
    print(f"📍 Parameters: {trainable_params:,} trainable / {total_params:,} total")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)


# =====================
# LOAD DATASET
# =====================
print("📂 Loading dataset...")
try:
    if not DATA_DIR.exists():
        print(f"❌ Data directory not found at {DATA_DIR}")
        sys.exit(1)
    
    dataset = BrainMRIDataset(
        data_dir=str(DATA_DIR),
        metadata_csv=str(METADATA_CSV),
        split='val',
        val_split=0.2,
        test_split=0.1,
        seed=42
    )
    print(f"✅ Dataset loaded: {len(dataset)} samples")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    sys.exit(1)


# =====================
# HELPER FUNCTIONS
# =====================
def run_inference(model, image_tensor):
    """Run model inference (argmax softmax only)."""
    with torch.no_grad():
        output = model(image_tensor.to(DEVICE))
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1)

    return pred.squeeze().cpu().numpy(), probs.squeeze().cpu().numpy()


# =====================
# GENERATE RESULTS
# =====================
print("\n" + "=" * 80)
print("GENERATING SEGMENTATION RESULTS")
print("=" * 80 + "\n")

# Select diverse samples
num_samples = min(ARGS.num_samples, len(dataset))
sample_indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

all_predictions = []
all_targets = []
all_metrics_list = []

for sample_num, idx in enumerate(sample_indices):
    print(f"Processing sample {sample_num + 1}/{num_samples} (index {idx})...")
    
    try:
        # Load sample
        img, mask = dataset[idx]
        
        # Run inference
        img_input = img.unsqueeze(0)
        pred_mask, probs = run_inference(model, img_input)
        
        # Compute metrics (shared helper)
        metrics_dict = compute_segmentation_metrics(
            torch.from_numpy(pred_mask).unsqueeze(0),
            torch.from_numpy(mask.numpy()).unsqueeze(0),
            num_classes=4,
        )
        
        all_predictions.append(pred_mask)
        all_targets.append(mask.numpy())
        all_metrics_list.append(metrics_dict)
        
        # Create visualization
        fig = plot_segmentation_results(
            image=img.numpy().transpose(1, 2, 0),
            mask=mask.numpy(),
            prediction=pred_mask,
            slice_idx=idx,
            save_path=SEG_DIR / f'sample_{idx:03d}.png'
        )
        plt.close(fig)
        
        print(f"  ✓ Dice: {metrics_dict['dice']:.4f}, IoU: {metrics_dict['iou']:.4f}")
        
    except Exception as e:
        print(f"  ❌ Error processing sample {idx}: {e}")
        continue

# =====================
# AGGREGATE METRICS
# =====================
print("\n" + "=" * 80)
print("GENERATING AGGREGATE VISUALIZATIONS")
print("=" * 80 + "\n")

try:
    # Concatenate all predictions and targets
    all_pred_concat = np.concatenate([p.flatten() for p in all_predictions])
    all_target_concat = np.concatenate([t.flatten() for t in all_targets])
    
    # Confusion matrix
    print("📊 Creating confusion matrix...")
    fig = plot_confusion_matrix(
        all_pred_concat, all_target_concat, num_classes=4, normalize=True,
        save_path=PLOTS_DIR / 'confusion_matrix.png'
    )
    plt.close(fig)
    
    # Class distribution
    print("📊 Creating class distribution...")
    fig = plot_class_distribution(
        all_target_concat, num_classes=4,
        save_path=PLOTS_DIR / 'class_distribution.png'
    )
    plt.close(fig)
    
    # Average metrics
    print("📊 Computing average metrics...")
    avg_metrics = {
        'dice': np.mean([m['dice'] for m in all_metrics_list]),
        'iou': np.mean([m['iou'] for m in all_metrics_list]),
        'accuracy': np.mean([m['accuracy'] for m in all_metrics_list]),
        'precision': np.mean([m['precision'] for m in all_metrics_list]),
        'recall': np.mean([m['recall'] for m in all_metrics_list]),
        'f1': np.mean([m['f1'] for m in all_metrics_list]),
        'dice_per_class': np.mean([m['dice_per_class'] for m in all_metrics_list], axis=0),
        'iou_per_class': np.mean([m['iou_per_class'] for m in all_metrics_list], axis=0),
        'precision_per_class': np.mean([m['precision_per_class'] for m in all_metrics_list], axis=0),
        'recall_per_class': np.mean([m['recall_per_class'] for m in all_metrics_list], axis=0),
    }
    
    # Per-class metrics
    print("📊 Creating per-class metrics...")
    fig = plot_per_class_metrics(
        avg_metrics,
        save_path=PLOTS_DIR / 'per_class_metrics.png'
    )
    plt.close(fig)
    
    # Metrics summary table
    print("📊 Creating metrics summary...")
    fig = create_metrics_summary_table(
        avg_metrics,
        save_path=PLOTS_DIR / 'metrics_summary.png'
    )
    plt.close(fig)

    # Inference speed benchmark using the same pipeline (softmax + argmax).
    sample_image, _ = dataset[int(sample_indices[0])]
    bench_input = sample_image.unsqueeze(0).to(DEVICE)
    iters = 20
    with torch.no_grad():
        _ = model(bench_input)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            output = model(bench_input)
            probs = torch.softmax(output, dim=1)
            _ = torch.argmax(probs, dim=1)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    inference_ms = ((t1 - t0) / iters) * 1000.0

    report_path = REPORTS_DIR / "metrics_summary.json"
    serializable = {
        "model_name": ARGS.model_name,
        "model_display_name": get_model_display_name(ARGS.model_name),
        'model_path': str(MODEL_PATH.resolve()),
        'data_dir': str(DATA_DIR.resolve()),
        'num_samples': num_samples,
        'device': str(DEVICE),
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "inference_ms_per_slice": float(inference_ms),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "val_split": 0.2,
        "test_split": 0.1,
        "eval_seed": 42,
        'dice': float(avg_metrics['dice']),
        'iou': float(avg_metrics['iou']),
        'accuracy': float(avg_metrics['accuracy']),
        'precision': float(avg_metrics['precision']),
        'recall': float(avg_metrics['recall']),
        'f1': float(avg_metrics['f1']),
        'dice_per_class': [float(x) for x in avg_metrics['dice_per_class']],
        'iou_per_class': [float(x) for x in avg_metrics['iou_per_class']],
        'precision_per_class': [float(x) for x in avg_metrics['precision_per_class']],
        'recall_per_class': [float(x) for x in avg_metrics['recall_per_class']],
        'class_names': ['Background', 'Necrotic', 'Edema', 'Enhancing'],
    }
    report_path.write_text(json.dumps(serializable, indent=2), encoding='utf-8')
    print(f"📄 Wrote {report_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nAggregated Metrics ({num_samples} samples):")
    print(f"  • Dice Score:  {avg_metrics['dice']:.4f}")
    print(f"  • IoU Score:   {avg_metrics['iou']:.4f}")
    print(f"  • Accuracy:    {avg_metrics['accuracy']:.4f}")
    print(f"  • Precision:   {avg_metrics['precision']:.4f}")
    print(f"  • Recall:      {avg_metrics['recall']:.4f}")
    print(f"  • F1-Score:    {avg_metrics['f1']:.4f}")
    
    print(f"\nPer-Class Dice Scores:")
    class_names = ['Background', 'Necrotic', 'Edema', 'Enhancing']
    for i, (name, dice) in enumerate(zip(class_names, avg_metrics['dice_per_class'])):
        print(f"  • {name}: {dice:.4f}")

except Exception as e:
    print(f"❌ Error creating aggregate visualizations: {e}")
    import traceback
    traceback.print_exc()

# =====================
# FINAL REPORT
# =====================
print("\n" + "=" * 80)
print("RESULTS SAVED")
print("=" * 80)
print(f"\n📁 Results folder: {RESULTS_DIR.absolute()}")
print(f"\n📄 Outputs:")
print(f"  • {SEG_DIR}/sample_*.png — per-sample segmentation panels")
print(f"  • {PLOTS_DIR}/ — confusion matrix, distributions, metric charts, summary table image")
print(f"  • {REPORTS_DIR}/metrics_summary.json — numeric summary for reporting")

print(f"\n✅ All results generated successfully!")
print("=" * 80)