"""
Ablation Study & Benchmarking for Brain MRI Segmentation
=========================================================
Generates REAL metrics: parameter counts, FLOPs estimates, inference timing,
component-wise analysis, and comparison charts.

Outputs saved to results/ablation/ as PNG charts + summary CSV.

Usage:
    python "ablation_study/ablation_study (1).py"
"""

import sys
import os
import time
import csv
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.architecture import AttentionUNet, ASPPModule, LightweightASPP, count_parameters


# ============ CONFIGURATION ============
RESULTS_DIR = Path('results/ablation')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cpu')  # CPU for fair timing comparison
INPUT_SHAPE = (1, 4, 128, 128)
NUM_TIMING_RUNS = 50
WARMUP_RUNS = 10


# ============ MODEL VARIANTS FOR ABLATION ============
VARIANTS = {
    'Full (Ours)': {
        'use_lightweight_aspp': True,
        'use_attention': True,
        'use_aspp': True,
        'description': 'MobileNetV2 + LightASPP + Attention Gates'
    },
    'Original ASPP': {
        'use_lightweight_aspp': False,
        'use_attention': True,
        'use_aspp': True,
        'description': 'MobileNetV2 + Heavy ASPP + Attention Gates'
    },
    'No Attention': {
        'use_lightweight_aspp': True,
        'use_attention': False,
        'use_aspp': True,
        'description': 'MobileNetV2 + LightASPP + Direct Skips'
    },
    'No ASPP': {
        'use_lightweight_aspp': True,
        'use_attention': True,
        'use_aspp': False,
        'description': 'MobileNetV2 + 1x1 Bottleneck + Attention Gates'
    },
    'Baseline': {
        'use_lightweight_aspp': True,
        'use_attention': False,
        'use_aspp': False,
        'description': 'MobileNetV2 + 1x1 Bottleneck + Direct Skips'
    },
}


def count_component_params(model):
    """Count parameters per component."""
    components = {}

    # Encoder
    enc_params = 0
    for name in ['enc0', 'enc1', 'enc2', 'enc3', 'enc4']:
        if hasattr(model, name):
            enc_params += sum(p.numel() for p in getattr(model, name).parameters())
    components['Encoder (MobileNetV2)'] = enc_params

    # Bottleneck
    if hasattr(model, 'bottleneck'):
        components['Bottleneck'] = sum(p.numel() for p in model.bottleneck.parameters())

    # Attention Gates
    attn_params = 0
    for name in ['attn0', 'attn1', 'attn2', 'attn3']:
        if hasattr(model, name):
            attn_params += sum(p.numel() for p in getattr(model, name).parameters())
    components['Attention Gates'] = attn_params

    # Decoder
    dec_params = 0
    for name in ['upconv0', 'upconv1', 'upconv2', 'upconv3',
                  'dec0', 'dec1', 'dec2', 'dec3']:
        if hasattr(model, name):
            dec_params += sum(p.numel() for p in getattr(model, name).parameters())
    components['Decoder'] = dec_params

    # Final Conv
    if hasattr(model, 'final_conv'):
        components['Output Layer'] = sum(p.numel() for p in model.final_conv.parameters())

    return components


def estimate_flops(model, input_shape):
    """Estimate FLOPs by hooking into conv/linear layers."""
    flops_list = []

    def hook_fn(module, inp, out):
        if isinstance(module, nn.Conv2d):
            _, c_in, h_in, w_in = inp[0].shape
            c_out, _, kh, kw = module.weight.shape
            groups = module.groups
            h_out, w_out = out.shape[2], out.shape[3]
            # MACs = c_out * h_out * w_out * (c_in/groups) * kh * kw
            macs = c_out * h_out * w_out * (c_in // groups) * kh * kw
            flops_list.append(macs * 2)  # FLOPs = 2 * MACs
        elif isinstance(module, nn.ConvTranspose2d):
            _, c_in, h_in, w_in = inp[0].shape
            c_out = module.out_channels
            kh, kw = module.kernel_size
            h_out, w_out = out.shape[2], out.shape[3]
            macs = c_in * h_out * w_out * c_out * kh * kw
            flops_list.append(macs * 2)
        elif isinstance(module, nn.Linear):
            macs = module.in_features * module.out_features
            flops_list.append(macs * 2)

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn))

    dummy = torch.randn(input_shape, device=DEVICE)
    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    return sum(flops_list)


def measure_inference_time(model, input_shape, num_runs=50, warmup=10):
    """Measure inference latency with warmup."""
    dummy = torch.randn(input_shape, device=DEVICE)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            model(dummy)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    return np.mean(times), np.std(times)


def measure_memory(model, input_shape):
    """Estimate peak memory usage."""
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    # Estimate activation memory (rough: 2x model size for forward pass)
    dummy = torch.randn(input_shape)
    input_mb = dummy.numel() * dummy.element_size() / 1e6
    estimated_peak = model_size_mb + input_mb * 10  # rough estimate
    return model_size_mb, estimated_peak


# ============ RUN ABLATION ============
def run_ablation():
    print("\n" + "=" * 80)
    print("ABLATION STUDY - Brain MRI Segmentation")
    print("=" * 80)

    results = []

    for name, config in VARIANTS.items():
        print(f"\n{'-' * 60}")
        print(f"Variant: {name}")
        print(f"Config:  {config['description']}")
        print(f"{'-' * 60}")

        # Build model
        model = AttentionUNet(
            in_channels=4, num_classes=4, pretrained=False,
            use_lightweight_aspp=config['use_lightweight_aspp'],
            use_attention=config['use_attention'],
            use_aspp=config['use_aspp']
        ).to(DEVICE)
        model.eval()

        # Count parameters
        total_params, trainable_params = count_parameters(model)
        components = count_component_params(model)

        # Estimate FLOPs
        flops = estimate_flops(model, INPUT_SHAPE)

        # Measure inference time
        mean_time, std_time = measure_inference_time(
            model, INPUT_SHAPE, NUM_TIMING_RUNS, WARMUP_RUNS
        )

        # Memory
        model_size_mb, peak_mb = measure_memory(model, INPUT_SHAPE)

        # Verify forward pass
        dummy = torch.randn(INPUT_SHAPE, device=DEVICE)
        with torch.no_grad():
            output = model(dummy)
        output_ok = output.shape == torch.Size([1, 4, 128, 128])

        # Print results
        print(f"  Total Params:    {total_params:>12,}")
        print(f"  Model Size:      {model_size_mb:>10.2f} MB")
        print(f"  FLOPs:           {flops / 1e9:>10.3f} GFLOPs")
        print(f"  Inference (CPU): {mean_time:>10.1f} ± {std_time:.1f} ms")
        print(f"  Output Shape OK: {'OK' if output_ok else 'FAIL'}")
        print(f"  Component Breakdown:")
        for comp_name, comp_params in components.items():
            pct = (comp_params / total_params * 100) if total_params > 0 else 0
            print(f"    {comp_name:25s}: {comp_params:>10,} ({pct:5.1f}%)")

        results.append({
            'name': name,
            'description': config['description'],
            'total_params': total_params,
            'model_size_mb': model_size_mb,
            'flops_gflops': flops / 1e9,
            'inference_ms': mean_time,
            'inference_std': std_time,
            'output_ok': output_ok,
            'components': components,
        })

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


# ============ GENERATE CHARTS ============
def generate_charts(results):
    """Generate professional comparison charts."""
    plt.style.use('seaborn-v0_8-whitegrid')

    names = [r['name'] for r in results]
    params = [r['total_params'] / 1e6 for r in results]
    flops = [r['flops_gflops'] for r in results]
    times = [r['inference_ms'] for r in results]
    time_stds = [r['inference_std'] for r in results]
    sizes = [r['model_size_mb'] for r in results]

    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']

    # ── Chart 1: Parameter Comparison ──
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, params, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_xlabel('Parameters (Millions)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study — Parameter Count Comparison', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, params):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}M', va='center', fontsize=11, fontweight='bold')
    ax.set_xlim(0, max(params) * 1.25)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'params_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved params_comparison.png")

    # ── Chart 2: FLOPs Comparison ──
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, flops, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_xlabel('GFLOPs', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study — Computational Cost (FLOPs)', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, flops):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=11, fontweight='bold')
    ax.set_xlim(0, max(flops) * 1.25)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'flops_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved flops_comparison.png")

    # ── Chart 3: Inference Time ──
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, times, xerr=time_stds, color=colors,
                   edgecolor='white', linewidth=1.5, capsize=4)
    ax.set_xlabel('Inference Time (ms) — CPU', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study — Inference Latency', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, times):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f} ms', va='center', fontsize=11, fontweight='bold')
    ax.set_xlim(0, max(times) * 1.35)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'inference_time.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved inference_time.png")

    # ── Chart 4: Component Breakdown (Pie chart for Full model) ──
    full_result = results[0]  # Full (Ours)
    comps = full_result['components']
    comp_names = list(comps.keys())
    comp_vals = list(comps.values())

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        comp_vals, labels=comp_names, autopct='%1.1f%%',
        colors=['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0'],
        startangle=90, pctdistance=0.8,
        wedgeprops=dict(linewidth=2, edgecolor='white')
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight('bold')
    ax.set_title(f'Component Breakdown — {full_result["name"]}\n({full_result["total_params"]:,} total params)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'component_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved component_breakdown.png")

    # ── Chart 5: Combined Summary (Multi-metric bar chart) ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Params
    axes[0].bar(range(len(names)), params, color=colors, edgecolor='white')
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8)
    axes[0].set_ylabel('Params (M)')
    axes[0].set_title('Parameters', fontweight='bold')

    # FLOPs
    axes[1].bar(range(len(names)), flops, color=colors, edgecolor='white')
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8)
    axes[1].set_ylabel('GFLOPs')
    axes[1].set_title('Computation', fontweight='bold')

    # Inference
    axes[2].bar(range(len(names)), times, color=colors, edgecolor='white')
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8)
    axes[2].set_ylabel('Time (ms)')
    axes[2].set_title('Latency (CPU)', fontweight='bold')

    fig.suptitle('Ablation Study — Multi-Metric Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'ablation_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved ablation_summary.png")

    # ── Chart 6: ASPP Comparison (Original vs Lightweight) ──
    aspp_original = results[1]  # Original ASPP variant
    aspp_light = results[0]     # Full (Ours) with LightASPP

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Params comparison
    labels = ['Original ASPP', 'Lightweight ASPP']
    aspp_params = [
        aspp_original['components'].get('Bottleneck', 0) / 1e6,
        aspp_light['components'].get('Bottleneck', 0) / 1e6
    ]
    bars = axes[0].bar(labels, aspp_params, color=['#F44336', '#2196F3'],
                       edgecolor='white', linewidth=2, width=0.5)
    axes[0].set_ylabel('Parameters (Millions)', fontweight='bold')
    axes[0].set_title('ASPP Bottleneck — Parameter Reduction', fontweight='bold')
    for bar, val in zip(bars, aspp_params):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f'{val:.2f}M', ha='center', fontsize=12, fontweight='bold')

    # Total model comparison
    total_params = [aspp_original['total_params'] / 1e6, aspp_light['total_params'] / 1e6]
    bars = axes[1].bar(labels, total_params, color=['#F44336', '#2196F3'],
                       edgecolor='white', linewidth=2, width=0.5)
    axes[1].set_ylabel('Parameters (Millions)', fontweight='bold')
    axes[1].set_title('Total Model — With vs Without Optimization', fontweight='bold')
    for bar, val in zip(bars, total_params):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{val:.2f}M', ha='center', fontsize=12, fontweight='bold')

    reduction = (1 - aspp_light['total_params'] / aspp_original['total_params']) * 100
    fig.suptitle(f'Lightweight ASPP Optimization — {reduction:.1f}% Total Param Reduction',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'aspp_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved aspp_comparison.png")


def save_csv(results):
    """Save results as CSV for reference."""
    csv_path = RESULTS_DIR / 'ablation_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Variant', 'Description', 'Total Params', 'Model Size (MB)',
            'GFLOPs', 'Inference (ms)', 'Inference Std (ms)'
        ])
        for r in results:
            writer.writerow([
                r['name'], r['description'], r['total_params'],
                f"{r['model_size_mb']:.2f}", f"{r['flops_gflops']:.3f}",
                f"{r['inference_ms']:.1f}", f"{r['inference_std']:.1f}"
            ])
    print("  Saved ablation_results.csv")


def print_summary_table(results):
    """Print formatted summary table."""
    print("\n" + "=" * 100)
    print("ABLATION STUDY - SUMMARY TABLE")
    print("=" * 100)
    header = f"{'Variant':<20} {'Params':>12} {'Size (MB)':>10} {'GFLOPs':>10} {'Latency (ms)':>14} {'Delta Params':>12}"
    print(header)
    print("-" * 100)

    baseline_params = results[0]['total_params']
    for r in results:
        delta = ((r['total_params'] - baseline_params) / baseline_params * 100)
        delta_str = f"{delta:+.1f}%" if r['name'] != results[0]['name'] else "baseline"
        print(f"{r['name']:<20} {r['total_params']:>12,} {r['model_size_mb']:>10.2f} "
              f"{r['flops_gflops']:>10.3f} {r['inference_ms']:>10.1f} ± {r['inference_std']:.1f} {delta_str:>12}")

    print("=" * 100)


# ============ MAIN ============
if __name__ == '__main__':
    results = run_ablation()

    print("\n" + "=" * 80)
    print("GENERATING CHARTS")
    print("=" * 80)
    generate_charts(results)
    save_csv(results)
    print_summary_table(results)

    print(f"\nAll results saved to: {RESULTS_DIR.absolute()}")
    print("Ablation study complete!")
