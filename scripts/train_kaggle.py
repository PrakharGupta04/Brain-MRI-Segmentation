
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

# CRITICAL: Kaggle-specific path handling
# Add parent directory to path for imports
sys.path.insert(0, '/kaggle/working')
if '/kaggle/working' not in sys.path:
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


class KaggleTrainer:
    """Kaggle-optimized training class"""
    
    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Kaggle always has GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Create output directory (Kaggle specific)
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
        """Build dataloaders (Kaggle paths)"""
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
    
    def train_epoch(self, model, train_loader, loss_fn, optimizer, scaler):
        """Train single epoch"""
        model.train()
        
        total_loss = 0.0
        metrics = SegmentationMetrics(num_classes=4)
        
        pbar = tqdm(train_loader, desc="Training", ncols=100)
        
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass with AMP (always on for Kaggle)
            optimizer.zero_grad()
            
            with autocast(enabled=True):
                outputs = model(images)
                loss, _ = loss_fn(outputs, masks)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
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
        
        # GradScaler always on for Kaggle (GPU available)
        scaler = GradScaler()
        
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
        description='Train brain MRI segmentation model (Kaggle)'
    )
    
    parser.add_argument('--data_dir', default='/kaggle/input/brats-processed/processed_data',
                       help='Path to processed_data/')
    parser.add_argument('--output_dir', default='/kaggle/working/outputs',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (GPU optimized)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--test_split', type=float, default=0.1, help='Test split')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--scheduler_step', type=int, default=30, help='Scheduler step size')
    parser.add_argument('--log_frequency', type=int, default=5, help='Log frequency')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create config (Kaggle specific defaults)
    config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'val_split': args.val_split,
        'test_split': args.test_split,
        'num_workers': args.num_workers,
        'early_stopping_patience': args.early_stopping_patience,
        'weight_decay': args.weight_decay,
        'scheduler_step': args.scheduler_step,
        'log_frequency': args.log_frequency,
        'seed': args.seed,
    }
    
    print("\n" + "=" * 80)
    print("BRAIN MRI SEGMENTATION - KAGGLE TRAINING")
    print("=" * 80)
    
    # Create trainer
    trainer = KaggleTrainer(config)
    
    # Build components
    model = trainer.build_model()
    train_loader, val_loader = trainer.build_dataloaders()
    optimizer, scheduler = trainer.build_optimizer_and_scheduler(model)
    loss_fn = CombinedLoss(num_classes=4, dice_weight=0.5, ce_weight=0.5)
    
    # Train
    trainer.train(model, train_loader, val_loader, loss_fn, optimizer, scheduler)
    
    print("\n" + "=" * 80)
    print("Training complete! Results saved to:", config['output_dir'])
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()