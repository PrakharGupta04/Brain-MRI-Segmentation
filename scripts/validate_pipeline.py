"""
Lightweight system validation for the Brain MRI segmentation framework.

This script is intentionally small and fast:
- no heavy training
- no large validation sweeps by default
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from models.model_registry import create_model, list_models, get_model_display_name
from models.metrics import SegmentationMetrics, compute_segmentation_metrics, count_model_parameters


def _print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def validate_registry():
    _print_header("1. Model registry integrity")
    keys = list(list_models())
    print(f"Available models ({len(keys)}): {', '.join(keys)}")
    for key in keys:
        model = create_model(key, in_channels=4, num_classes=4, pretrained=False)
        total = sum(p.numel() for p in model.parameters())
        print(f"  - {key}: {get_model_display_name(key)} ({total:,} params)")


def validate_forward_pass(device: torch.device):
    _print_header("2. Forward pass shape sanity check")
    x = torch.randn(1, 4, 128, 128, device=device)
    for key in list_models():
        model = create_model(key, in_channels=4, num_classes=4, pretrained=False).to(device)
        with torch.no_grad():
            y = model(x)
        print(f"  - {key}: output shape {tuple(y.shape)}")
        assert y.shape == (1, 4, 128, 128), f"{key} produced incorrect shape {y.shape}"


def validate_metrics_consistency():
    _print_header("3. Metrics consistency (Dice vs F1)")
    num_classes = 4
    preds = torch.randint(0, num_classes, (2, 128, 128))
    targets = torch.randint(0, num_classes, (2, 128, 128))

    metrics = SegmentationMetrics(num_classes=num_classes)
    metrics.update(preds, targets)
    m = metrics.compute_all_metrics()

    dice = m["dice"]
    f1 = m["f1"]
    print(f"Macro Dice: {dice:.4f}, Macro F1: {f1:.4f}")
    assert abs(dice - f1) < 1e-5, "Dice and F1 should match for segmentation"


def validate_checkpoint_loading():
    _print_header("4. Checkpoint loading (if present)")
    root = Path(".")
    default_ckpt = root / "outputs" / "best_model.pth"
    if not default_ckpt.exists():
        print("No default checkpoint at outputs/best_model.pth — skipping.")
        return

    ckpt = torch.load(default_ckpt, map_location="cpu")
    model = create_model("mobilenet_attention_unet", in_channels=4, num_classes=4, pretrained=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    total = count_model_parameters(model)["total"]
    print(f"Loaded checkpoint into mobilenet_attention_unet ({total:,} params)")


def validate_reports():
    _print_header("5. Existing report sanity check (if any)")
    base = Path("results/experiments")
    if not base.exists():
        print("No experiment folder at results/experiments — skipping.")
        return

    any_found = False
    for report_path in base.glob("*/**/reports/metrics_summary.json"):
        any_found = True
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  - {report_path}: FAILED to read JSON ({e})")
            continue

        required = ["dice", "iou", "precision", "recall", "f1", "num_samples"]
        missing = [k for k in required if k not in data]
        if missing:
            print(f"  - {report_path}: missing fields {missing}")
        else:
            print(
                f"  - {report_path}: ok | Dice={data['dice']:.4f}, "
                f"IoU={data['iou']:.4f}, F1={data['f1']:.4f}, "
                f"samples={data['num_samples']}"
            )
    if not any_found:
        print("No metrics_summary.json files found under results/experiments.")


def validate_tiny_inference(device: torch.device):
    _print_header("6. Tiny inference sanity test")
    model = create_model("mobilenet_attention_unet", in_channels=4, num_classes=4, pretrained=False).to(device)
    x = torch.randn(1, 4, 128, 128, device=device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    y = torch.randint(0, 4, (1, 128, 128), device=device)
    metrics = compute_segmentation_metrics(preds, y, num_classes=4)
    print(
        f"Softmax+argmax inference ok | "
        f"Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}, F1={metrics['f1']:.4f}"
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    validate_registry()
    validate_forward_pass(device)
    validate_metrics_consistency()
    validate_checkpoint_loading()
    validate_reports()
    validate_tiny_inference(device)

    _print_header("Validation complete")


if __name__ == "__main__":
    main()

