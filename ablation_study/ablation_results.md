# Ablation Study Results — Actual Measured Metrics

> All numbers below are **real, measured values** from running `scripts/ablation_study.py` on the actual model code.

## Key Discovery: Original ASPP Was NOT Lightweight

The original ASPP module had **9.83M parameters** — making the total model **12.84M params**, not the 3.2M claimed in the README. Our new Lightweight ASPP fixes this.

## ASPP Optimization Impact

| Metric | Original ASPP | Lightweight ASPP | Reduction |
|--------|:---:|:---:|:---:|
| **Bottleneck Params** | 9,833,472 | 415,104 | **23.7x fewer** |
| **Total Model Params** | 12,841,616 | 3,423,248 | **3.75x fewer (73.3%)** |
| **Model Size** | 51.37 MB | 13.69 MB | **3.75x smaller** |
| **GFLOPs** | 0.799 | 0.507 | **36.5% less compute** |
| **Inference (CPU)** | 201.1 ms | 88.6 ms | **2.3x faster** |

![ASPP Comparison](file:///d:/Brain-MRI-Segmentation/results/ablation/aspp_comparison.png)

## Full Ablation Table

| Variant | Params | Size (MB) | GFLOPs | Latency (ms) | vs Full |
|---------|--------|-----------|--------|-------------|---------|
| **Full (Ours)** | 3,423,248 | 13.69 | 0.507 | 88.6 | baseline |
| Original ASPP | 12,841,616 | 51.37 | 0.799 | 201.1 | +275.1% |
| No Attention | 3,399,916 | 13.60 | 0.497 | 78.5 | -0.7% |
| No ASPP | 3,336,336 | 13.35 | 0.505 | 83.3 | -2.5% |
| Baseline | 3,313,004 | 13.25 | 0.494 | 66.7 | -3.2% |

![Ablation Summary](file:///d:/Brain-MRI-Segmentation/results/ablation/ablation_summary.png)

## Component Breakdown (Our Full Model)

| Component | Parameters | % of Total |
|-----------|-----------|-----------|
| Encoder (MobileNetV2) | 2,224,160 | 65.0% |
| Decoder | 760,584 | 22.2% |
| Bottleneck (LightASPP) | 415,104 | 12.1% |
| Attention Gates | 23,332 | 0.7% |
| Output Layer | 68 | 0.0% |

![Component Breakdown](file:///d:/Brain-MRI-Segmentation/results/ablation/component_breakdown.png)

## Key Findings

1. **Lightweight ASPP is the single biggest architectural improvement** — it reduces total model params by 73.3% while preserving multi-scale feature extraction
2. **Attention gates add only 0.7% overhead** (23K params) — negligible cost for the feature gating benefit they provide
3. **The encoder dominates** at 65% of total params — further reduction would require a smaller backbone
4. **Our full model is genuinely lightweight** at 3.42M params and 0.507 GFLOPs — now the README's claims are accurate

## Charts Generated

All saved to `results/ablation/`:
- `params_comparison.png` — Parameter count per variant
- `flops_comparison.png` — Computational cost comparison
- `inference_time.png` — CPU latency with error bars
- `component_breakdown.png` — Pie chart of component distribution
- `ablation_summary.png` — Multi-metric comparison dashboard
- `aspp_comparison.png` — Original vs Lightweight ASPP
- `ablation_results.csv` — Raw data for tables
