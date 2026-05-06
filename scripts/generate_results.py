# ============ SCRIPTS/GENERATE_RESULTS.PY - COMPLETE CORRECTED VERSION ============

"""
Generate segmentation results on validation set samples
Saves all visualizations to results/ folder
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.architecture import AttentionUNet
from utils.dataset_loader import BrainMRIDataset
from utils.visualization import (
    plot_segmentation_results, 
    plot_confusion_matrix,
    plot_class_distribution,
    plot_per_class_metrics,
    create_metrics_summary_table
)
from models.metrics import SegmentationMetrics

# =====================
# CONFIGURATION
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path('outputs/best_model.pth')
DATA_DIR = Path('E:/Brain_MRI_DL/processed_data')
METADATA_CSV = DATA_DIR / 'metadata.csv'
RESULTS_DIR = Path('results')

# Create results folder
RESULTS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("BRAIN MRI SEGMENTATION - RESULTS GENERATION")
print("=" * 80)
print(f"\n📍 Device: {DEVICE}")
print(f"📍 Model: {MODEL_PATH}")
print(f"📍 Data: {DATA_DIR}")
print(f"📍 Results: {RESULTS_DIR}\n")


# =====================
# LOAD MODEL
# =====================
print("📂 Loading model...")
try:
    model = AttentionUNet(in_channels=4, num_classes=4, pretrained=False)
    
    if not MODEL_PATH.exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        sys.exit(1)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully")
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
def normalize_image(img):
    """Normalize image to 0-1 range"""
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min == 0:
        return np.zeros_like(img)
    return (img - img_min) / (img_max - img_min)


def run_inference(model, image_tensor):
    """Run model inference"""
    with torch.no_grad():
        output = model(image_tensor.to(DEVICE))
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1)
        
        # Apply threshold for edema class
        tumor_mask = probs[:, 2, :, :] > 0.01
        pred[tumor_mask] = 2
        
    return pred.squeeze().cpu().numpy(), probs.squeeze().cpu().numpy()


# =====================
# GENERATE RESULTS
# =====================
print("\n" + "=" * 80)
print("GENERATING SEGMENTATION RESULTS")
print("=" * 80 + "\n")

# Select diverse samples
num_samples = min(10, len(dataset))
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
        
        # Compute metrics
        metrics = SegmentationMetrics(num_classes=4)
        metrics.update(torch.from_numpy(pred_mask).unsqueeze(0), 
                      torch.from_numpy(mask.numpy()).unsqueeze(0))
        metrics_dict = metrics.compute_all_metrics()
        
        all_predictions.append(pred_mask)
        all_targets.append(mask.numpy())
        all_metrics_list.append(metrics_dict)
        
        # Prepare image for display
        image_display = img[0].numpy()  # T1 weighted
        image_display = normalize_image(image_display)
        
        # Create visualization
        fig = plot_segmentation_results(
            image=img.numpy().transpose(1, 2, 0),
            mask=mask.numpy(),
            prediction=pred_mask,
            slice_idx=idx,
            save_path=RESULTS_DIR / f'sample_{idx:03d}.png'
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
        save_path=RESULTS_DIR / 'confusion_matrix.png'
    )
    plt.close(fig)
    
    # Class distribution
    print("📊 Creating class distribution...")
    fig = plot_class_distribution(
        all_target_concat, num_classes=4,
        save_path=RESULTS_DIR / 'class_distribution.png'
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
        save_path=RESULTS_DIR / 'per_class_metrics.png'
    )
    plt.close(fig)
    
    # Metrics summary table
    print("📊 Creating metrics summary...")
    fig = create_metrics_summary_table(
        avg_metrics,
        save_path=RESULTS_DIR / 'metrics_summary.png'
    )
    plt.close(fig)
    
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
print(f"\n📄 Files generated:")
print(f"  • sample_*.png (individual segmentation results)")
print(f"  • confusion_matrix.png (confusion matrix)")
print(f"  • class_distribution.png (class distribution)")
print(f"  • per_class_metrics.png (per-class metrics)")
print(f"  • metrics_summary.png (metrics summary table)")

print(f"\n✅ All results generated successfully!")
print("=" * 80)