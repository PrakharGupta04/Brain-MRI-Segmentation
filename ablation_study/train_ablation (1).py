"""
Ablation Study — Training Script
==================================
Trains all 5 model variants on the SAME data with IDENTICAL hyperparameters.
Saves checkpoints for each variant to outputs/ablation/.

Variants:
  1. Full (Ours):    MobileNetV2 + LightASPP + Attention Gates
  2. Original ASPP:  MobileNetV2 + Heavy ASPP + Attention Gates
  3. No Attention:   MobileNetV2 + LightASPP + Direct Skips
  4. No ASPP:        MobileNetV2 + 1x1 Bottleneck + Attention Gates
  5. Baseline:       MobileNetV2 + 1x1 Bottleneck + Direct Skips

Usage:
    python "ablation_study/train_ablation (1).py" --data_dir E:/Brain_MRI_DL/processed_data_filtered_v1 --epochs 15
    python "ablation_study/train_ablation (1).py" --data_dir /kaggle/input/processed_data_filtered_v1 --epochs 15 --batch_size 16

To train a SINGLE variant (useful for Kaggle time limits):
    python "ablation_study/train_ablation (1).py" --data_dir ... --variant full
    python "ablation_study/train_ablation (1).py" --data_dir ... --variant no_attention
"""

import sys
import os
import argparse
import time
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.architecture import AttentionUNet, count_parameters
from models.losses import CombinedLoss
from models.metrics import SegmentationMetrics
from utils.dataset_loader import BrainMRIDataModule
from utils.transforms import get_augmentation_pipeline


# ============ VARIANT DEFINITIONS ============
VARIANTS = {
    'full': {
        'use_lightweight_aspp': True,
        'use_attention': True,
        'use_aspp': True,
        'label': 'Full (Ours)',
        'description': 'MobileNetV2 + LightASPP + Attention Gates'
    },
    'original_aspp': {
        'use_lightweight_aspp': False,
        'use_attention': True,
        'use_aspp': True,
        'label': 'Original ASPP',
        'description': 'MobileNetV2 + Heavy ASPP + Attention Gates'
    },
    'no_attention': {
        'use_lightweight_aspp': True,
        'use_attention': False,
        'use_aspp': True,
        'label': 'No Attention',
        'description': 'MobileNetV2 + LightASPP + Direct Skips'
    },
    'no_aspp': {
        'use_lightweight_aspp': True,
        'use_attention': True,
        'use_aspp': False,
        'label': 'No ASPP',
        'description': 'MobileNetV2 + 1x1 Bottleneck + Attention Gates'
    },
    'baseline': {
        'use_lightweight_aspp': True,
        'use_attention': False,
        'use_aspp': False,
        'label': 'Baseline',
        'description': 'MobileNetV2 + 1x1 Bottleneck + Direct Skips'
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description='Ablation Study Training')

    # Data
    parser.add_argument('--data_dir', type=str, default=r"E:\Brain_MRI_DL\processed_data_filtered_v1",
                        help='Path to processed_data directory')
    parser.add_argument('--output_dir', type=str, default='outputs/ablation',
                        help='Output directory for checkpoints and logs')

    # Training
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs per variant (default: 30)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers (default: 4)')
    parser.add_argument('--dice_weight', type=float, default=0.5)
    parser.add_argument('--ce_weight', type=float, default=0.5)
    parser.add_argument('--use_class_weights', action='store_true', default=True)
    parser.add_argument('--no_class_weights', action='store_true',
                        help='Disable class-weighted loss')

    # Variant selection
    parser.add_argument('--variant', type=str, default='all',
                        choices=['all'] + list(VARIANTS.keys()),
                        help='Train a specific variant or all (default: all)')

    # Hardware
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable AMP')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    return parser.parse_args()


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_dataloaders(args):
    """Build train/val dataloaders (SAME for all variants)."""
    data_dir = args.data_dir
    metadata_csv = os.path.join(data_dir, 'metadata.csv')

    dm = BrainMRIDataModule(
        data_dir=data_dir,
        metadata_csv=metadata_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.2,
        test_split=0.1,
        transforms_train=get_augmentation_pipeline('train'),
        transforms_val=get_augmentation_pipeline('val'),
        seed=args.seed,
    )
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    print(f"  Train samples: {len(dm._train_dataset)}")
    print(f"  Val samples:   {len(dm._val_dataset)}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")

    return train_loader, val_loader


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp):
    """Train for one epoch. Returns average loss and metrics."""
    model.train()
    total_loss = 0.0
    metrics = SegmentationMetrics(num_classes=4)

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(images)
                loss, loss_dict = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss, loss_dict = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

        # Track metrics
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            metrics.update(preds, masks)

    avg_loss = total_loss / len(train_loader)
    metrics_dict = metrics.compute_all_metrics()

    return avg_loss, metrics_dict


@torch.no_grad()
def validate(model, val_loader, criterion, device, use_amp):
    """Validate. Returns average loss and metrics."""
    model.eval()
    total_loss = 0.0
    metrics = SegmentationMetrics(num_classes=4)

    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)

        if use_amp:
            with autocast():
                outputs = model(images)
                loss, loss_dict = criterion(outputs, masks)
        else:
            outputs = model(images)
            loss, loss_dict = criterion(outputs, masks)

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        metrics.update(preds, masks)

    avg_loss = total_loss / len(val_loader)
    metrics_dict = metrics.compute_all_metrics()

    return avg_loss, metrics_dict


def train_variant(variant_key, variant_config, train_loader, val_loader, args):
    """Train a single model variant end-to-end."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = args.amp and not args.no_amp and torch.cuda.is_available()

    variant_dir = Path(args.output_dir) / variant_key
    variant_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"TRAINING: {variant_config['label']}")
    print(f"Config:   {variant_config['description']}")
    print(f"Device:   {device} | AMP: {use_amp}")
    print(f"Epochs:   {args.epochs} | LR: {args.lr} | Batch: {args.batch_size}")
    print(f"Save to:  {variant_dir}")
    print(f"{'=' * 70}")

    # Build model
    set_seed(args.seed)  # Reset seed before each variant for fair comparison
    model = AttentionUNet(
        in_channels=4, num_classes=4, pretrained=True,
        use_lightweight_aspp=variant_config['use_lightweight_aspp'],
        use_attention=variant_config['use_attention'],
        use_aspp=variant_config['use_aspp']
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"  Model size: {total_params * 4 / 1e6:.2f} MB (FP32)")

    # Optimizer, scheduler, loss — IDENTICAL for all variants
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(args.epochs // 3, 5), gamma=0.5)
    use_class_weights = args.use_class_weights and not args.no_class_weights
    class_weights = torch.tensor([0.05, 1.0, 2.0, 2.0], dtype=torch.float32) if use_class_weights else None
    criterion = CombinedLoss(
        num_classes=4,
        dice_weight=args.dice_weight,
        ce_weight=args.ce_weight,
        dice_class_weights=class_weights,
        ce_class_weights=class_weights,
    )
    scaler = GradScaler() if use_amp else None

    # Training loop
    best_val_dice = 0.0
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )

        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, use_amp
        )

        # Step scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log
        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_dice': train_metrics['dice'],
            'val_dice': val_metrics['dice'],
            'train_iou': train_metrics['iou'],
            'val_iou': val_metrics['iou'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time,
        }

        # Add per-class dice
        for i, name in enumerate(['bg', 'ncr', 'ed', 'et']):
            row[f'val_dice_{name}'] = val_metrics['dice_per_class'][i]
            row[f'val_iou_{name}'] = val_metrics['iou_per_class'][i]

        history.append(row)

        # Print progress
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"Dice: {train_metrics['dice']:.4f}/{val_metrics['dice']:.4f} | "
              f"IoU: {val_metrics['iou']:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Time: {epoch_time:.1f}s")

        # Save best model
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_metrics['dice'],
                'val_iou': val_metrics['iou'],
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'variant': variant_key,
                'config': variant_config,
                'args': vars(args),
                'total_params': total_params,
            }
            torch.save(checkpoint, variant_dir / 'best_model.pth')
            print(f"  >> Saved best model (Dice: {best_val_dice:.4f})")

    total_time = time.time() - start_time

    # Save last model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'val_dice': val_metrics['dice'],
        'val_iou': val_metrics['iou'],
        'variant': variant_key,
        'total_params': total_params,
    }, variant_dir / 'last_model.pth')

    # Save training history CSV
    csv_path = variant_dir / 'training_history.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    # Save variant summary
    summary = {
        'variant': variant_key,
        'label': variant_config['label'],
        'description': variant_config['description'],
        'total_params': total_params,
        'best_val_dice': best_val_dice,
        'best_val_iou': float(checkpoint['val_iou']),
        'final_val_loss': float(val_loss),
        'total_time_seconds': total_time,
        'epochs': args.epochs,
        'config': variant_config,
    }
    with open(variant_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  DONE: {variant_config['label']}")
    print(f"  Best Val Dice: {best_val_dice:.4f}")
    print(f"  Total Time:    {total_time / 60:.1f} minutes")
    print(f"  Files saved:   {variant_dir}")

    # Cleanup GPU memory
    del model, optimizer, scheduler, criterion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


# ============ MAIN ============
def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("ABLATION STUDY — TRAINING PIPELINE")
    print("=" * 70)
    print(f"Data:       {args.data_dir}")
    print(f"Output:     {args.output_dir}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seed:       {args.seed}")

    # Determine which variants to train
    if args.variant == 'all':
        variants_to_train = VARIANTS
    else:
        variants_to_train = {args.variant: VARIANTS[args.variant]}

    print(f"Variants:   {list(variants_to_train.keys())}")
    print("=" * 70)

    # Build dataloaders ONCE (shared across all variants)
    print("\nLoading dataset...")
    set_seed(args.seed)
    train_loader, val_loader = build_dataloaders(args)

    # Train each variant
    all_summaries = []
    for variant_key, variant_config in variants_to_train.items():
        summary = train_variant(
            variant_key, variant_config,
            train_loader, val_loader, args
        )
        all_summaries.append(summary)

    # Print final comparison
    print("\n" + "=" * 90)
    print("ABLATION STUDY — FINAL COMPARISON")
    print("=" * 90)
    print(f"{'Variant':<20} {'Params':>12} {'Best Dice':>12} {'Best IoU':>12} {'Time (min)':>12}")
    print("-" * 90)
    for s in all_summaries:
        print(f"{s['label']:<20} {s['total_params']:>12,} {s['best_val_dice']:>12.4f} "
              f"{s['best_val_iou']:>12.4f} {s['total_time_seconds']/60:>12.1f}")
    print("=" * 90)

    # Save combined summary
    output_dir = Path(args.output_dir)
    with open(output_dir / 'all_summaries.json', 'w') as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nAll results saved to: {output_dir.absolute()}")
    print('Next step: run  python "ablation_study/evaluate_ablation (1).py"  to generate charts')


if __name__ == '__main__':
    main()
