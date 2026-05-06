# ============ UTILS/VISUALIZATION.PY - CORRECTED VERSION ============

"""
Visualization utilities for segmentation analysis with real saving
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from pathlib import Path


def mri_window_to_uint8(slice_2d: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """
    Robust MRI display scaling: percentile window reduces noise from outliers
    compared to min-max normalization. Returns uint8 [0, 255].
    """
    x = np.asarray(slice_2d, dtype=np.float64)
    flat = x.ravel()
    if flat.size == 0:
        return np.zeros_like(slice_2d, dtype=np.uint8)
    lo, hi = np.percentile(flat, (p_low, p_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi <= lo:
            return np.zeros_like(slice_2d, dtype=np.uint8)
    clipped = np.clip(x, lo, hi)
    out = (clipped - lo) / (hi - lo)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def plot_confusion_matrix(predictions, targets, num_classes=4, normalize=True, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        predictions: Predicted class indices
        targets: Ground truth labels
        num_classes: Number of classes
        normalize: Whether to normalize
        save_path: Path to save figure
    
    Returns:
        fig: Matplotlib figure
    """
    # Flatten
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Compute confusion matrix
    cm = confusion_matrix(target_flat, pred_flat, labels=range(num_classes))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    class_names = ['Background', 'Necrotic', 'Edema', 'Enhancing']
    
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', ax=ax, cbar_kws={'label': 'Count'},
                xticklabels=class_names[:num_classes],
                yticklabels=class_names[:num_classes])
    
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''), 
                fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved confusion matrix to {save_path}")
    
    return fig


def plot_class_distribution(targets, num_classes=4, save_path=None):
    """
    Plot class distribution
    
    Args:
        targets: Ground truth labels
        num_classes: Number of classes
        save_path: Path to save figure
    
    Returns:
        fig: Matplotlib figure
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Count pixels per class
    unique, counts = np.unique(targets.flatten(), return_counts=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    class_names = ['Background', 'Necrotic', 'Edema', 'Enhancing']
    colors = ['#808080', '#ff4444', '#44ff44', '#4444ff']
    
    class_counts = np.zeros(num_classes)
    for u, c in zip(unique, counts):
        if u < num_classes:
            class_counts[u] = c
    
    percentages = class_counts / class_counts.sum() * 100
    
    bars = ax.bar(class_names[:num_classes], percentages, color=colors[:num_classes])
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_title('Class Distribution in Dataset', fontweight='bold', fontsize=14)
    ax.set_ylim([0, 100])
    
    # Add percentage labels
    for i, (name, pct) in enumerate(zip(class_names[:num_classes], percentages)):
        ax.text(i, pct + 2, f'{pct:.1f}%', ha='center', fontweight='bold', fontsize=10)
    
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved class distribution to {save_path}")
    
    return fig


def plot_per_class_metrics(metrics_dict, class_names=None, save_path=None):
    """
    Plot per-class metrics
    
    Args:
        metrics_dict: Dictionary with per-class metrics
        class_names: Names of classes
        save_path: Path to save figure
    
    Returns:
        fig: Matplotlib figure
    """
    if class_names is None:
        class_names = ['Background', 'Necrotic', 'Edema', 'Enhancing']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = ['dice_per_class', 'iou_per_class', 'precision_per_class', 'recall_per_class']
    titles = ['Dice Score', 'IoU Score', 'Precision', 'Recall']
    colors = ['#808080', '#ff4444', '#44ff44', '#4444ff']
    
    for idx, (ax, metric_key, title) in enumerate(zip(axes.flatten(), metrics_to_plot, titles)):
        if metric_key in metrics_dict:
            values = metrics_dict[metric_key]
            if isinstance(values, torch.Tensor):
                values = values.cpu().numpy()
            
            bars = ax.bar(class_names, values, color=colors)
            ax.set_ylabel(title, fontweight='bold')
            ax.set_title(f'{title} per Class', fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(values):
                ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)
    
    plt.suptitle('Per-Class Segmentation Metrics', fontweight='bold', fontsize=14, y=1.00)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved per-class metrics to {save_path}")
    
    return fig


def plot_segmentation_results(image, mask, prediction, slice_idx=0, save_path=None):
    """
    Plot segmentation results side-by-side
    
    Args:
        image: Input MRI [H, W, C]
        mask: Ground truth mask [H, W]
        prediction: Predicted mask [H, W]
        slice_idx: Slice index for title
        save_path: Path to save figure
    
    Returns:
        fig: Matplotlib figure
    """
    # Convert torch tensors if needed
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    
    # Window first channel (or 2D) for clearer grayscale MRI
    if image.ndim == 3:
        img_display = mri_window_to_uint8(image[:, :, 0]).astype(np.float64) / 255.0
    else:
        img_display = mri_window_to_uint8(image).astype(np.float64) / 255.0
    
    # Colors for segmentation
    colors = ['black', '#ff4444', '#44ff44', '#4444ff']
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    
    # Original
    axes[0].imshow(img_display, cmap='gray')
    axes[0].set_title('Original MRI', fontweight='bold')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(img_display, cmap='gray')
    axes[1].imshow(mask, cmap=cmap, alpha=0.6, vmin=0, vmax=3)
    axes[1].set_title('Ground Truth', fontweight='bold')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(img_display, cmap='gray')
    axes[2].imshow(prediction, cmap=cmap, alpha=0.6, vmin=0, vmax=3)
    axes[2].set_title('Prediction', fontweight='bold')
    axes[2].axis('off')
    
    # Overlay difference
    axes[3].imshow(img_display, cmap='gray')
    # Show prediction in color, overlay with GT
    difference = np.where(mask == prediction, 0, 1)
    error_mask = np.zeros((*difference.shape, 3), dtype=np.uint8)
    error_mask[difference == 1] = [255, 0, 0]  # Red for errors
    axes[3].imshow(error_mask, alpha=0.5)
    axes[3].set_title('Error Map (Red=Mismatch)', fontweight='bold')
    axes[3].axis('off')
    
    plt.suptitle(f'Sample {slice_idx} - Segmentation Analysis', fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved segmentation results to {save_path}")
    
    return fig


def create_metrics_summary_table(metrics_dict, class_names=None, save_path=None):
    """
    Create a summary table of all metrics
    
    Args:
        metrics_dict: Dictionary with metrics
        class_names: Class names
        save_path: Path to save figure
    
    Returns:
        fig: Matplotlib figure
    """
    if class_names is None:
        class_names = ['Background', 'Necrotic', 'Edema', 'Enhancing']
    
    # Extract per-class metrics
    dice = metrics_dict.get('dice_per_class', [])
    iou = metrics_dict.get('iou_per_class', [])
    prec = metrics_dict.get('precision_per_class', [])
    rec = metrics_dict.get('recall_per_class', [])
    
    if any(isinstance(m, torch.Tensor) for m in [dice, iou, prec, rec]):
        dice = dice.cpu().numpy() if isinstance(dice, torch.Tensor) else np.array(dice)
        iou = iou.cpu().numpy() if isinstance(iou, torch.Tensor) else np.array(iou)
        prec = prec.cpu().numpy() if isinstance(prec, torch.Tensor) else np.array(prec)
        rec = rec.cpu().numpy() if isinstance(rec, torch.Tensor) else np.array(rec)
    
    # Create figure with table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    table_data = []
    table_data.append(['Class', 'Dice', 'IoU', 'Precision', 'Recall', 'F1-Score'])
    
    for i, class_name in enumerate(class_names):
        if len(dice) > i and len(iou) > i and len(prec) > i and len(rec) > i:
            f1 = 2 * (prec[i] * rec[i]) / (prec[i] + rec[i] + 1e-6)
            table_data.append([
                class_name,
                f'{dice[i]:.4f}',
                f'{iou[i]:.4f}',
                f'{prec[i]:.4f}',
                f'{rec[i]:.4f}',
                f'{f1:.4f}',
            ])
    
    # Add macro average
    if len(dice) > 0:
        table_data.append([
            'Average',
            f'{metrics_dict.get("dice", 0):.4f}',
            f'{metrics_dict.get("iou", 0):.4f}',
            f'{metrics_dict.get("precision", 0):.4f}',
            f'{metrics_dict.get("recall", 0):.4f}',
            f'{metrics_dict.get("f1", 0):.4f}',
        ])
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.15, 0.15, 0.15, 0.2, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Color header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#0066cc')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color last row (average)
    for i in range(len(table_data[0])):
        table[(len(table_data) - 1, i)].set_facecolor('#e3f2fd')
        table[(len(table_data) - 1, i)].set_text_props(weight='bold')
    
    # Alternate row colors
    for i in range(1, len(table_data) - 1):
        color = '#f8f9fa' if i % 2 == 0 else '#ffffff'
        for j in range(len(table_data[0])):
            table[(i, j)].set_facecolor(color)
    
    plt.title('Segmentation Metrics Summary', fontweight='bold', fontsize=14, pad=20)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved metrics summary to {save_path}")
    
    return fig


if __name__ == "__main__":
    print("Testing visualization utilities...")
    
    # Create dummy data
    predictions = np.random.randint(0, 4, (128, 128))
    targets = np.random.randint(0, 4, (128, 128))
    
    # Create results folder
    Path('results').mkdir(exist_ok=True)
    
    # Test confusion matrix
    fig = plot_confusion_matrix(predictions, targets, save_path='results/confusion_matrix.png')
    print("✓ Confusion matrix plot created")
    plt.close(fig)
    
    # Test class distribution
    fig = plot_class_distribution(targets, save_path='results/class_distribution.png')
    print("✓ Class distribution plot created")
    plt.close(fig)
    
    # Test per-class metrics
    metrics = {
        'dice_per_class': np.array([0.96, 0.68, 0.82, 0.79]),
        'iou_per_class': np.array([0.92, 0.51, 0.70, 0.66]),
        'precision_per_class': np.array([0.98, 0.72, 0.85, 0.81]),
        'recall_per_class': np.array([0.95, 0.64, 0.79, 0.77]),
        'dice': 0.81,
        'iou': 0.70,
        'precision': 0.84,
        'recall': 0.79,
        'f1': 0.81,
    }
    
    fig = plot_per_class_metrics(metrics, save_path='results/per_class_metrics.png')
    print("✓ Per-class metrics plot created")
    plt.close(fig)
    
    # Test segmentation results
    image = np.random.randn(128, 128, 4)
    mask = np.random.randint(0, 4, (128, 128))
    prediction = np.random.randint(0, 4, (128, 128))
    
    fig = plot_segmentation_results(image, mask, prediction, slice_idx=0, 
                                   save_path='results/sample_0.png')
    print("✓ Segmentation results plot created")
    plt.close(fig)
    
    # Test metrics summary
    fig = create_metrics_summary_table(metrics, save_path='results/metrics_summary.png')
    print("✓ Metrics summary created")
    plt.close(fig)
    
    print("\n✅ All visualization utilities tested successfully!")
    print("📁 Results saved in /results folder")