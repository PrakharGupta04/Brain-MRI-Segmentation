# ============ SCRIPTS/TRAIN_LOCAL.PY - FULLY CORRECTED (FINE-TUNE PATCH) ============

"""
Training Script for Brain MRI Segmentation (Local/Small GPU)

Features:
- Mixed Precision Training (AMP)
- Learning Rate Scheduling
- Early Stopping
- Checkpoint Saving
- Metrics Logging
- Visualization

FINE-TUNE PATCH (class-imbalance fix):
- Added --resume_checkpoint to load a saved best_model.pth before training
- Added --finetune flag that caps epochs=5 and lowers LR to 1e-4
- CombinedLoss now accepts class_weights so background (class 0) is down-weighted
  and tumor classes (1,2,3) are up-weighted: [0.05, 1.0, 2.0, 2.0]
- class_weights are applied independently to both the Dice and CrossEntropy
  components inside CombinedLoss (no changes to architecture, dataset, or optimizer)

ORIGINAL FIXES (unchanged):
1. All tqdm formatting uses f-strings: pbar.set_postfix({'loss': f"{loss.item():.4f}"})
2. Removed unused imports (compute_class_weights)
3. Consistent parameter counting (count_parameters from architecture)
4. All imports use modular paths
5. No nested f-string syntax errors
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# CRITICAL: Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import with modular paths
from models.architecture import AttentionUNet, count_parameters
from models.losses import CombinedLoss
from models.metrics import SegmentationMetrics, count_model_parameters
from utils.dataset_loader import BrainMRIDataModule
from utils.transforms import get_augmentation_pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Main training class"""
    
    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(
            'cuda' if config['device'] == 'gpu' and torch.cuda.is_available() else 'cpu'
        )
        
        logger.info(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Config saved to {config_path}")
        
        # Initialize tracking
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'train_iou': [],
            'val_iou': [],
            'lr': [],
        }
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def build_model(self):
        """Build model"""
        logger.info("Building model...")
        model = AttentionUNet(
            in_channels=4,
            num_classes=4,
            pretrained=True
        )
        model = model.to(self.device)
        
        # Print model summary
        total, trainable = count_parameters(model)
        logger.info(f"Model Parameters: {trainable:,} / {total:,}")
        logger.info(f"Model Size (FP32): {total * 4 / 1e6:.2f} MB")
        
        return model
    
    def build_dataloaders(self):
        """Build dataloaders"""
        logger.info("Building dataloaders...")
        
        # Get augmentation pipelines
        transforms_train = get_augmentation_pipeline(stage='train')
        transforms_val = get_augmentation_pipeline(stage='val')
        
        # Create data module
        dm = BrainMRIDataModule(
            data_dir=self.config['data_dir'],
            metadata_csv=f"{self.config['data_dir']}/metadata.csv",
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            val_split=self.config['val_split'],
            test_split=self.config['test_split'],
            transforms_train=transforms_train,
            transforms_val=transforms_val,
            seed=self.config['seed']
        )
        
        dm.setup()
        
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        
        logger.info(f"Train samples: {len(dm._train_dataset)}")
        logger.info(f"Val samples: {len(dm._val_dataset)}")
        logger.info(f"Batches per epoch: {len(train_loader)}")
        
        return train_loader, val_loader
    
    def build_optimizer_and_scheduler(self, model):
        """Build optimizer and scheduler"""
        logger.info("Building optimizer and scheduler...")
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config['scheduler_step'],
            gamma=0.5
        )
        
        return optimizer, scheduler

    # ------------------------------------------------------------------
    # FINE-TUNE PATCH: resume from an existing checkpoint
    # ------------------------------------------------------------------
    def load_checkpoint(self, model, optimizer, checkpoint_path: str):
        """
        Load weights (and optionally optimizer state) from a saved checkpoint.
        Returns the epoch number stored in the checkpoint so callers know
        where training left off (used only for logging; epoch counter always
        restarts from 1 for the fine-tune run).
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. "
                "Pass the correct path via --resume_checkpoint."
            )

        logger.info(f"Loading checkpoint from {ckpt_path} ...")
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("  ✓ Model weights loaded.")

        # Restore optimizer state so momentum / adaptive terms carry over,
        # but the LR will be overwritten by the fine-tune config value below.
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Override the LR stored in the checkpoint with the fine-tune LR
            for pg in optimizer.param_groups:
                pg['lr'] = self.config['learning_rate']
            logger.info(
                f"  ✓ Optimizer state loaded. LR overridden to "
                f"{self.config['learning_rate']}"
            )

        original_epoch = checkpoint.get('epoch', '?')
        logger.info(
            f"  ✓ Checkpoint was saved at epoch {original_epoch}. "
            f"Fine-tuning will run for {self.config['epochs']} additional epoch(s)."
        )

        # Carry over the best val loss so the very first fine-tune epoch can
        # only save a new best_model.pth if it actually improves.
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
            logger.info(f"  ✓ Restored best_val_loss = {self.best_val_loss:.4f}")

        return original_epoch
    # ------------------------------------------------------------------

    def train_epoch(self, model, train_loader, loss_fn, optimizer, scaler):
        """Train single epoch"""
        model.train()
        
        total_loss = 0.0
        metrics = SegmentationMetrics(num_classes=4)
        
        pbar = tqdm(train_loader, desc="Training", ncols=100)
        
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass with AMP
            optimizer.zero_grad()
            
            with autocast(enabled=self.config['use_amp']):
                outputs = model(images)
                loss, _ = loss_fn(outputs, masks)
            
            # Backward pass
            if self.config['use_amp']:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            metrics.update(outputs.detach(), masks)
            
            # FIXED: Proper f-string formatting for tqdm
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        dice = metrics.compute_dice().item()
        iou = metrics.compute_iou().item()
        
        return avg_loss, dice, iou
    
    def validate_epoch(self, model, val_loader, loss_fn):
        """Validate single epoch"""
        model.eval()
        
        total_loss = 0.0
        metrics = SegmentationMetrics(num_classes=4)
        
        pbar = tqdm(val_loader, desc="Validation", ncols=100)
        
        with torch.no_grad():
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = model(images)
                loss, _ = loss_fn(outputs, masks)
                
                # Update metrics
                total_loss += loss.item()
                metrics.update(outputs, masks)
                
                # FIXED: Proper f-string formatting for tqdm
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Compute epoch metrics
        avg_loss = total_loss / len(val_loader)
        dice = metrics.compute_dice().item()
        iou = metrics.compute_iou().item()
        
        return avg_loss, dice, iou
    
    def save_checkpoint(self, model, optimizer, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,   # persist so future resumes work
            'config': self.config,
        }
        
        # Save last checkpoint
        last_path = self.output_dir / 'last_model.pth'
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def plot_metrics(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        axes[0, 0].plot(self.history['epoch'], self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['epoch'], self.history['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Dice Score
        axes[0, 1].plot(self.history['epoch'], self.history['train_dice'], label='Train')
        axes[0, 1].plot(self.history['epoch'], self.history['val_dice'], label='Val')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].set_title('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # IoU Score
        axes[1, 0].plot(self.history['epoch'], self.history['train_iou'], label='Train')
        axes[1, 0].plot(self.history['epoch'], self.history['val_iou'], label='Val')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU Score')
        axes[1, 0].set_title('IoU Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(self.history['epoch'], self.history['lr'], 'g-')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'training_metrics.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved metrics plot to {plot_path}")
        plt.close()
    
    def train(self, model, train_loader, val_loader, loss_fn, optimizer, scheduler):
        """Main training loop"""
        logger.info("Starting training...")
        
        if self.config['use_amp']:
            scaler = GradScaler()
        else:
            scaler = None
        
        for epoch in range(1, self.config['epochs'] + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{self.config['epochs']}")
            logger.info(f"{'='*60}")
            
            # Train
            train_loss, train_dice, train_iou = self.train_epoch(
                model, train_loader, loss_fn, optimizer, scaler
            )
            
            # Validate
            val_loss, val_dice, val_iou = self.validate_epoch(
                model, val_loader, loss_fn
            )
            
            # Update scheduler
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            
            # Log metrics
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_dice'].append(train_dice)
            self.history['val_dice'].append(val_dice)
            self.history['train_iou'].append(train_iou)
            self.history['val_iou'].append(val_iou)
            self.history['lr'].append(current_lr)
            
            # FIXED: Proper f-string formatting
            logger.info(
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(model, optimizer, epoch, is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= self.config['early_stopping_patience']:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
            
            # Save metrics periodically
            if epoch % self.config['log_frequency'] == 0:
                self.save_history()
        
        # Save final history and plots
        self.save_history()
        self.plot_metrics()
        
        logger.info("Training complete!")
    
    def save_history(self):
        """Save training history to CSV"""
        df = pd.DataFrame(self.history)
        csv_path = self.output_dir / 'training_logs.csv'
        df.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description='Train brain MRI segmentation model'
    )
    
    parser.add_argument('--data_dir', required=True, help='Path to processed_data/')
    parser.add_argument('--output_dir', default='outputs/train_local', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--test_split', type=float, default=0.1, help='Test split')
    parser.add_argument('--device', choices=['gpu', 'cpu'], default='gpu', help='Device')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--use_amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--scheduler_step', type=int, default=20, help='Scheduler step size')
    parser.add_argument('--log_frequency', type=int, default=5, help='Log frequency')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # ------------------------------------------------------------------
    # FINE-TUNE PATCH: two new arguments
    # ------------------------------------------------------------------
    parser.add_argument(
        '--resume_checkpoint',
        type=str,
        default=None,
        help=(
            'Path to a saved checkpoint (.pth) to resume from. '
            'Example: outputs/train_local/best_model.pth'
        ),
    )
    parser.add_argument(
        '--finetune',
        action='store_true',
        default=False,
        help=(
            'Fine-tune mode: caps epochs to 5, lowers LR to 1e-4, '
            'and enables class-weighted loss [0.05, 1.0, 2.0, 2.0] '
            'to fix background-bias / tumor under-detection.'
        ),
    )
    # ------------------------------------------------------------------

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # FINE-TUNE PATCH: apply fine-tune overrides before building config
    # ------------------------------------------------------------------
    if args.finetune:
        if args.epochs == 50:          # only override if user left the default
            args.epochs = 5
        if args.learning_rate == 0.001:  # only override if user left the default
            args.learning_rate = 1e-4
        logger.info(
            f"[Fine-tune mode] epochs={args.epochs}, lr={args.learning_rate}"
        )
    # ------------------------------------------------------------------

    # Create config
    config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'val_split': args.val_split,
        'test_split': args.test_split,
        'device': args.device,
        'num_workers': args.num_workers,
        'use_amp': args.use_amp,
        'early_stopping_patience': args.early_stopping_patience,
        'weight_decay': args.weight_decay,
        'scheduler_step': args.scheduler_step,
        'log_frequency': args.log_frequency,
        'seed': args.seed,
    }
    
    print("\n" + "=" * 80)
    print("BRAIN MRI SEGMENTATION - LOCAL TRAINING")
    print("=" * 80)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Build components
    model = trainer.build_model()
    train_loader, val_loader = trainer.build_dataloaders()
    optimizer, scheduler = trainer.build_optimizer_and_scheduler(model)

    # ------------------------------------------------------------------
    # FINE-TUNE PATCH: class-weighted CombinedLoss
    #
    # Weight rationale
    # ----------------
    # class 0  (background) → 0.05  very low; model already predicts this well
    # class 1  (necrotic core / NCR) → 1.0   moderate tumour class
    # class 2  (peritumoral oedema / ED) → 2.0  often missed; penalise harder
    # class 3  (GD-enhancing tumour / ET) → 2.0  often missed; penalise harder
    #
    # These weights are passed as a torch.Tensor to CombinedLoss so that:
    #   • CrossEntropyLoss receives them via its `weight` parameter
    #   • DiceLoss uses them to scale per-class Dice terms before averaging
    #
    # If --finetune is NOT set the weights default to None (uniform), which
    # reproduces the original behaviour exactly.
    # ------------------------------------------------------------------
    if args.finetune:
        class_weights = torch.tensor(
            [0.05, 1.0, 2.0, 2.0], dtype=torch.float32
        )
        logger.info(
            f"[Fine-tune mode] Using class weights: {class_weights.tolist()}"
        )
    else:
        class_weights = None   # uniform weights → original behaviour
    # ------------------------------------------------------------------

    loss_fn = CombinedLoss(
        num_classes=4,
        dice_weight=0.5,
        ce_weight=0.5,
        dice_class_weights=class_weights, 
        ce_class_weights=class_weights,
    )

    # ------------------------------------------------------------------
    # FINE-TUNE PATCH: load checkpoint weights before training
    # ------------------------------------------------------------------
    if args.resume_checkpoint is not None:
        trainer.load_checkpoint(model, optimizer, args.resume_checkpoint)
    # ------------------------------------------------------------------

    # Train
    trainer.train(model, train_loader, val_loader, loss_fn, optimizer, scheduler)
    
    print("\n" + "=" * 80)
    print("Training complete! Results saved to:", config['output_dir'])
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()