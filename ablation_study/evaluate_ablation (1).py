"""
Ablation Study — Evaluation & Chart Generation
=================================================
Loads trained checkpoints for all 5 variants and generates:
  - Complete comparison table (Params + FLOPs + Dice + IoU)
  - Per-class Dice comparison chart
  - Training curves overlay (all variants)
  - Efficiency vs Accuracy scatter plot
  - Component breakdown pie charts
  - Summary CSV for papers/reports

Usage (after training):
    python "ablation_study/evaluate_ablation (1).py"
    python "ablation_study/evaluate_ablation (1).py" --ablation_dir outputs/ablation --results_dir results/ablation
"""

import sys
import os
import json
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.architecture import AttentionUNet, count_parameters
from models.metrics import SegmentationMetrics


# ============ CONFIG ============
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate ablation study results')
    parser.add_argument('--ablation_dir', type=str, default='outputs/ablation',
                        help='Directory containing trained variant checkpoints')
    parser.add_argument('--results_dir', type=str, default='results/ablation',
                        help='Output directory for charts and tables')
    return parser.parse_args()


VARIANT_ORDER = ['full', 'original_aspp', 'no_attention', 'no_aspp', 'baseline']
VARIANT_LABELS = {
    'full': 'Full (Ours)',
    'original_aspp': 'Original ASPP',
    'no_attention': 'No Attention',
    'no_aspp': 'No ASPP',
    'baseline': 'Baseline',
}
VARIANT_COLORS = {
    'full': '#2196F3',
    'original_aspp': '#F44336',
    'no_attention': '#4CAF50',
    'no_aspp': '#FF9800',
    'baseline': '#9C27B0',
}
CLASS_NAMES = ['Background', 'Necrotic', 'Edema', 'Enhancing']


def estimate_flops(model, device='cpu'):
    """Estimate FLOPs via forward hooks."""
    flops_list = []

    def hook_fn(module, inp, out):
        if isinstance(module, nn.Conv2d):
            _, c_in, _, _ = inp[0].shape
            c_out, _, kh, kw = module.weight.shape
            h_out, w_out = out.shape[2], out.shape[3]
            macs = c_out * h_out * w_out * (c_in // module.groups) * kh * kw
            flops_list.append(macs * 2)
        elif isinstance(module, nn.ConvTranspose2d):
            _, c_in, _, _ = inp[0].shape
            c_out = module.out_channels
            kh, kw = module.kernel_size
            h_out, w_out = out.shape[2], out.shape[3]
            macs = c_in * h_out * w_out * c_out * kh * kw
            flops_list.append(macs * 2)

    hooks = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(m.register_forward_hook(hook_fn))

    dummy = torch.randn(1, 4, 128, 128, device=device)
    with torch.no_grad():
        model(dummy)
    for h in hooks:
        h.remove()
    return sum(flops_list)


def measure_inference_time(model, device='cpu', runs=30, warmup=5):
    """Measure inference latency."""
    dummy = torch.randn(1, 4, 128, 128, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if device != 'cpu' and torch.cuda.is_available():
            torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(runs):
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(dummy)
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
    return np.mean(times), np.std(times)


def load_variant_data(ablation_dir):
    """Load all variant summaries and training histories."""
    variants = {}
    ablation_path = Path(ablation_dir)

    for key in VARIANT_ORDER:
        variant_dir = ablation_path / key
        if not variant_dir.exists():
            print(f"  WARNING: {key} not found at {variant_dir}, skipping")
            continue

        data = {'key': key, 'label': VARIANT_LABELS[key]}

        # Load summary
        summary_path = variant_dir / 'summary.json'
        if summary_path.exists():
            with open(summary_path) as f:
                data['summary'] = json.load(f)
        else:
            print(f"  WARNING: No summary.json for {key}")
            continue

        # Load training history
        history_path = variant_dir / 'training_history.csv'
        if history_path.exists():
            data['history'] = pd.read_csv(history_path)

        # Load best checkpoint metrics
        best_path = variant_dir / 'best_model.pth'
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location='cpu')
            data['best_metrics'] = checkpoint.get('val_metrics', {})
            data['best_epoch'] = checkpoint.get('epoch', 0)
            data['config'] = checkpoint.get('config', {})
        else:
            print(f"  WARNING: No best_model.pth for {key}")
            continue

        variants[key] = data

    return variants


def compute_efficiency_metrics(variants):
    """Compute FLOPs and inference time for each variant."""
    device = 'cpu'

    for key, data in variants.items():
        config = data.get('config', {})
        model = AttentionUNet(
            in_channels=4, num_classes=4, pretrained=False,
            use_lightweight_aspp=config.get('use_lightweight_aspp', True),
            use_attention=config.get('use_attention', True),
            use_aspp=config.get('use_aspp', True)
        ).to(device)
        model.eval()

        total_params, _ = count_parameters(model)
        flops = estimate_flops(model, device)
        latency_mean, latency_std = measure_inference_time(model, device)

        data['total_params'] = total_params
        data['model_size_mb'] = total_params * 4 / 1e6
        data['flops_gflops'] = flops / 1e9
        data['latency_ms'] = latency_mean
        data['latency_std'] = latency_std

        del model

    return variants


# ============ CHART GENERATORS ============

def chart_accuracy_comparison(variants, results_dir):
    """Bar chart comparing Dice and IoU across variants."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    keys = [k for k in VARIANT_ORDER if k in variants]
    labels = [VARIANT_LABELS[k] for k in keys]
    colors = [VARIANT_COLORS[k] for k in keys]

    # Dice
    dice_vals = [variants[k]['summary']['best_val_dice'] for k in keys]
    bars = axes[0].bar(labels, dice_vals, color=colors, edgecolor='white', linewidth=1.5)
    axes[0].set_ylabel('Dice Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Validation Dice Score', fontsize=13, fontweight='bold')
    axes[0].set_ylim(0, 1)
    for bar, val in zip(bars, dice_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')

    # IoU
    iou_vals = [variants[k]['summary']['best_val_iou'] for k in keys]
    bars = axes[1].bar(labels, iou_vals, color=colors, edgecolor='white', linewidth=1.5)
    axes[1].set_ylabel('IoU Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Validation IoU Score', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, 1)
    for bar, val in zip(bars, iou_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')

    plt.suptitle('Ablation Study - Accuracy Comparison', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(results_dir / 'accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved accuracy_comparison.png")


def chart_per_class_dice(variants, results_dir):
    """Grouped bar chart of per-class Dice scores."""
    keys = [k for k in VARIANT_ORDER if k in variants]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(CLASS_NAMES))
    width = 0.15
    offsets = np.arange(len(keys)) - len(keys) / 2 + 0.5

    for i, key in enumerate(keys):
        metrics = variants[key].get('best_metrics', {})
        per_class = metrics.get('dice_per_class', [0, 0, 0, 0])
        if hasattr(per_class, 'tolist'):
            per_class = per_class.tolist() if hasattr(per_class, 'tolist') else list(per_class)
        bars = ax.bar(x + offsets[i] * width, per_class, width,
                      label=VARIANT_LABELS[key], color=VARIANT_COLORS[key],
                      edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dice Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Dice Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(results_dir / 'per_class_dice.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved per_class_dice.png")


def chart_training_curves(variants, results_dir):
    """Overlay training curves for all variants."""
    keys = [k for k in VARIANT_ORDER if k in variants and 'history' in variants[k]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for key in keys:
        h = variants[key]['history']
        color = VARIANT_COLORS[key]
        label = VARIANT_LABELS[key]

        # Train Loss
        axes[0, 0].plot(h['epoch'], h['train_loss'], color=color, label=label, linewidth=1.5)
        # Val Loss
        axes[0, 1].plot(h['epoch'], h['val_loss'], color=color, label=label, linewidth=1.5)
        # Val Dice
        axes[1, 0].plot(h['epoch'], h['val_dice'], color=color, label=label, linewidth=1.5)
        # Val IoU
        axes[1, 1].plot(h['epoch'], h['val_iou'], color=color, label=label, linewidth=1.5)

    titles = ['Training Loss', 'Validation Loss', 'Validation Dice', 'Validation IoU']
    ylabels = ['Loss', 'Loss', 'Dice Score', 'IoU Score']
    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('Ablation Study - Training Curves', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(results_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved training_curves.png")


def chart_efficiency_vs_accuracy(variants, results_dir):
    """Scatter plot: Parameters vs Dice (bubble size = FLOPs)."""
    keys = [k for k in VARIANT_ORDER if k in variants]

    fig, ax = plt.subplots(figsize=(10, 7))

    for key in keys:
        v = variants[key]
        params_m = v['total_params'] / 1e6
        dice = v['summary']['best_val_dice']
        gflops = v['flops_gflops']
        color = VARIANT_COLORS[key]
        label = VARIANT_LABELS[key]

        ax.scatter(params_m, dice, s=gflops * 500, c=color, alpha=0.7,
                   edgecolors='black', linewidth=1, zorder=5)
        ax.annotate(label, (params_m, dice), textcoords="offset points",
                    xytext=(10, 5), fontsize=9, fontweight='bold')

    ax.set_xlabel('Parameters (Millions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best Validation Dice', fontsize=12, fontweight='bold')
    ax.set_title('Efficiency vs Accuracy Trade-off\n(bubble size = GFLOPs)',
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(results_dir / 'efficiency_vs_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved efficiency_vs_accuracy.png")


def chart_full_comparison_table(variants, results_dir):
    """Generate a matplotlib table image for papers."""
    keys = [k for k in VARIANT_ORDER if k in variants]

    headers = ['Variant', 'Params', 'Size(MB)', 'GFLOPs', 'Latency(ms)', 'Dice', 'IoU']
    rows = []
    for key in keys:
        v = variants[key]
        rows.append([
            VARIANT_LABELS[key],
            f"{v['total_params']:,}",
            f"{v['model_size_mb']:.1f}",
            f"{v['flops_gflops']:.3f}",
            f"{v['latency_ms']:.1f}",
            f"{v['summary']['best_val_dice']:.4f}",
            f"{v['summary']['best_val_iou']:.4f}",
        ])

    fig, ax = plt.subplots(figsize=(14, 2 + len(rows) * 0.5))
    ax.axis('off')

    table = ax.table(cellText=rows, colLabels=headers, loc='center',
                     cellLoc='center', colColours=['#E3F2FD'] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    # Bold the best values
    ax.set_title('Ablation Study - Complete Comparison', fontsize=14,
                 fontweight='bold', pad=20)

    plt.tight_layout()
    fig.savefig(results_dir / 'comparison_table.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved comparison_table.png")


def save_final_csv(variants, results_dir):
    """Save complete results as CSV."""
    keys = [k for k in VARIANT_ORDER if k in variants]
    csv_path = results_dir / 'complete_ablation_results.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Variant', 'Description', 'Total Params', 'Model Size (MB)',
            'GFLOPs', 'Latency (ms)', 'Best Dice', 'Best IoU',
            'Dice BG', 'Dice NCR', 'Dice ED', 'Dice ET',
            'Training Time (min)'
        ])
        for key in keys:
            v = variants[key]
            metrics = v.get('best_metrics', {})
            per_class = metrics.get('dice_per_class', [0, 0, 0, 0])
            if hasattr(per_class, 'tolist'):
                per_class = list(per_class)

            writer.writerow([
                VARIANT_LABELS[key],
                v['summary']['description'],
                v['total_params'],
                f"{v['model_size_mb']:.1f}",
                f"{v['flops_gflops']:.3f}",
                f"{v['latency_ms']:.1f}",
                f"{v['summary']['best_val_dice']:.4f}",
                f"{v['summary']['best_val_iou']:.4f}",
                f"{per_class[0]:.4f}" if len(per_class) > 0 else "N/A",
                f"{per_class[1]:.4f}" if len(per_class) > 1 else "N/A",
                f"{per_class[2]:.4f}" if len(per_class) > 2 else "N/A",
                f"{per_class[3]:.4f}" if len(per_class) > 3 else "N/A",
                f"{v['summary']['total_time_seconds']/60:.1f}",
            ])
    print(f"  Saved complete_ablation_results.csv")


# ============ MAIN ============
def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("ABLATION STUDY - EVALUATION & CHART GENERATION")
    print("=" * 70)
    print(f"Checkpoints: {args.ablation_dir}")
    print(f"Output:      {args.results_dir}")

    # Load all variant data
    print("\nLoading variant data...")
    variants = load_variant_data(args.ablation_dir)

    if not variants:
        print("\nERROR: No trained variants found!")
        print(f"Expected directories in {args.ablation_dir}:")
        for key in VARIANT_ORDER:
            print(f"  {args.ablation_dir}/{key}/best_model.pth")
        print("\nRun train_ablation.py first.")
        return

    print(f"  Loaded {len(variants)} variants: {list(variants.keys())}")

    # Compute efficiency metrics
    print("\nComputing efficiency metrics...")
    variants = compute_efficiency_metrics(variants)

    # Print summary
    print("\n" + "=" * 100)
    print("COMPLETE ABLATION RESULTS")
    print("=" * 100)
    print(f"{'Variant':<20} {'Params':>12} {'GFLOPs':>10} {'Latency':>12} {'Dice':>10} {'IoU':>10}")
    print("-" * 100)
    for key in VARIANT_ORDER:
        if key not in variants:
            continue
        v = variants[key]
        print(f"{VARIANT_LABELS[key]:<20} {v['total_params']:>12,} {v['flops_gflops']:>10.3f} "
              f"{v['latency_ms']:>9.1f} ms {v['summary']['best_val_dice']:>10.4f} "
              f"{v['summary']['best_val_iou']:>10.4f}")
    print("=" * 100)

    # Generate all charts
    print("\nGenerating charts...")
    chart_accuracy_comparison(variants, results_dir)
    chart_per_class_dice(variants, results_dir)
    chart_training_curves(variants, results_dir)
    chart_efficiency_vs_accuracy(variants, results_dir)
    chart_full_comparison_table(variants, results_dir)
    save_final_csv(variants, results_dir)

    print(f"\nAll results saved to: {results_dir.absolute()}")
    print("Done!")


if __name__ == '__main__':
    main()
