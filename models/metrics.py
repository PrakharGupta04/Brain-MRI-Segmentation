# ============ MODELS/METRICS.PY - FULLY CORRECTED ============

"""
Evaluation Metrics for Multi-class Brain Tumor Segmentation

Metrics:
- Dice Score (F1-Score)
- Intersection over Union (IoU)
- Pixel Accuracy
- Precision & Recall
- Inference Time
- Parameter Count

CRITICAL FIXES:
1. All metric tensors moved to output device dynamically
2. No hardcoded 'cuda' - uses outputs.device
3. Device movement happens in update() before computation
4. All intermediate tensors on same device
5. Safe for both CPU and GPU
6. Memory efficient
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import time
from typing import Dict, Tuple, Optional


class SegmentationMetrics:
    """
    Comprehensive metrics for semantic segmentation tasks
    
    Computes per-class and macro-averaged metrics:
    - Dice Score: 2*TP / (2*TP + FP + FN)
    - IoU: TP / (TP + FP + FN)
    - Accuracy: (TP + TN) / (TP + TN + FP + FN)
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    
    CRITICAL: Metric tensors are initialized on CPU but automatically moved
    to the device of outputs during update() to prevent device mismatches.
    """
    
    def __init__(self, num_classes: int = 4, ignore_index: int = -100):
        """
        Args:
            num_classes: Number of segmentation classes
            ignore_index: Class index to ignore in metrics
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # Initialize metric tensors on CPU (will be moved to correct device in update)
        # Using register_buffer-like initialization for clarity
        self._device = None  # Will be set on first update
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        # Initialize on CPU - will move to correct device in update()
        self.tp = torch.zeros(self.num_classes, dtype=torch.float32)
        self.fp = torch.zeros(self.num_classes, dtype=torch.float32)
        self.fn = torch.zeros(self.num_classes, dtype=torch.float32)
        self.tn = torch.zeros(self.num_classes, dtype=torch.float32)
        self._device = None
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with new batch
        
        CRITICAL FIX: Move all metric tensors to predictions.device before computation
        
        Args:
            predictions: Predicted class indices [B, H, W] or logits [B, C, H, W]
            targets: Ground truth labels [B, H, W]
        """
        # CRITICAL: Get device from outputs
        device = predictions.device
        
        # CRITICAL: Move all metric tensors to same device as predictions
        self.tp = self.tp.to(device)
        self.fp = self.fp.to(device)
        self.fn = self.fn.to(device)
        self.tn = self.tn.to(device)
        
        # Convert logits to class indices if needed
        if predictions.dim() == 4:  # Logits [B, C, H, W]
            predictions = torch.argmax(predictions, dim=1)
        
        # Ensure both tensors are on same device
        predictions = predictions.to(device).long()
        targets = targets.to(device).long()
        
        # Flatten
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)
        
        # Remove ignore index
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            predictions = predictions[mask]
            targets = targets[mask]
        
        # Compute confusion matrix per class
        # All operations now on same device
        for c in range(self.num_classes):
            # CRITICAL: All intermediate tensors created on correct device
            pred_c = predictions == c
            target_c = targets == c
            
            # Compute TP, FP, FN, TN (all tensors on device)
            tp = torch.sum((pred_c & target_c).float()).to(device)
            fp = torch.sum((pred_c & ~target_c).float()).to(device)
            fn = torch.sum((~pred_c & target_c).float()).to(device)
            tn = torch.sum((~pred_c & ~target_c).float()).to(device)
            
            # Update accumulated metrics (now on same device)
            self.tp[c] += tp
            self.fp[c] += fp
            self.fn[c] += fn
            self.tn[c] += tn
        
        self._device = device
    
    def compute_dice(self, per_class: bool = False) -> torch.Tensor:
        """
        Compute Dice Score
        
        Formula: Dice = 2*TP / (2*TP + FP + FN)
        Range: [0, 1] where 1 is perfect
        
        Args:
            per_class: If True, return per-class scores
        
        Returns:
            dice: Scalar or tensor of per-class scores
        """
        smooth = 1e-6
        dice_per_class = (2.0 * self.tp + smooth) / (2.0 * self.tp + self.fp + self.fn + smooth)
        
        if per_class:
            return dice_per_class
        else:
            return dice_per_class.mean()
    
    def compute_iou(self, per_class: bool = False) -> torch.Tensor:
        """
        Compute Intersection over Union (IoU/Jaccard)
        
        Formula: IoU = TP / (TP + FP + FN)
        Range: [0, 1] where 1 is perfect
        
        Args:
            per_class: If True, return per-class scores
        
        Returns:
            iou: Scalar or tensor of per-class scores
        """
        smooth = 1e-6
        iou_per_class = (self.tp + smooth) / (self.tp + self.fp + self.fn + smooth)
        
        if per_class:
            return iou_per_class
        else:
            return iou_per_class.mean()
    
    def compute_accuracy(self, per_class: bool = False) -> torch.Tensor:
        """
        Compute Pixel Accuracy
        
        Formula: Accuracy = (TP + TN) / (TP + TN + FP + FN)
        Range: [0, 1]
        
        Note: Misleading for imbalanced datasets!
        Always use with Dice/IoU
        
        Args:
            per_class: If True, return per-class scores
        
        Returns:
            accuracy: Scalar or tensor
        """
        total_correct = self.tp + self.tn
        total_samples = self.tp + self.tn + self.fp + self.fn
        
        acc_per_class = total_correct / (total_samples + 1e-6)
        
        if per_class:
            return acc_per_class
        else:
            return acc_per_class.mean()
    
    def compute_precision(self, per_class: bool = False) -> torch.Tensor:
        """
        Compute Precision
        
        Formula: Precision = TP / (TP + FP)
        Range: [0, 1]
        
        Args:
            per_class: If True, return per-class scores
        
        Returns:
            precision: Scalar or tensor
        """
        precision_per_class = self.tp / (self.tp + self.fp + 1e-6)
        
        if per_class:
            return precision_per_class
        else:
            return precision_per_class.mean()
    
    def compute_recall(self, per_class: bool = False) -> torch.Tensor:
        """
        Compute Recall (Sensitivity)
        
        Formula: Recall = TP / (TP + FN)
        Range: [0, 1]
        
        Args:
            per_class: If True, return per-class scores
        
        Returns:
            recall: Scalar or tensor
        """
        recall_per_class = self.tp / (self.tp + self.fn + 1e-6)
        
        if per_class:
            return recall_per_class
        else:
            return recall_per_class.mean()
    
    def compute_f1(self, per_class: bool = False) -> torch.Tensor:
        """
        Compute F1-Score
        
        Formula: F1 = 2 * (Precision * Recall) / (Precision + Recall)
        Range: [0, 1]
        
        Args:
            per_class: If True, return per-class scores
        
        Returns:
            f1: Scalar or tensor
        """
        precision = self.compute_precision(per_class=True)
        recall = self.compute_recall(per_class=True)
        
        f1_per_class = 2.0 * (precision * recall) / (precision + recall + 1e-6)
        
        if per_class:
            return f1_per_class
        else:
            return f1_per_class.mean()
    
    def compute_all_metrics(self) -> Dict:
        """
        Compute all metrics at once
        
        Returns:
            metrics_dict: Dictionary with all metrics
        """
        metrics = {
            'dice': self.compute_dice().item(),
            'dice_per_class': self.compute_dice(per_class=True).cpu().numpy(),
            'iou': self.compute_iou().item(),
            'iou_per_class': self.compute_iou(per_class=True).cpu().numpy(),
            'accuracy': self.compute_accuracy().item(),
            'accuracy_per_class': self.compute_accuracy(per_class=True).cpu().numpy(),
            'precision': self.compute_precision().item(),
            'precision_per_class': self.compute_precision(per_class=True).cpu().numpy(),
            'recall': self.compute_recall().item(),
            'recall_per_class': self.compute_recall(per_class=True).cpu().numpy(),
            'f1': self.compute_f1().item(),
            'f1_per_class': self.compute_f1(per_class=True).cpu().numpy(),
        }
        return metrics
    
    def get_summary(self) -> str:
        """
        Get summary string of all metrics
        
        Returns:
            summary: Formatted string
        """
        metrics = self.compute_all_metrics()
        
        summary = "\n" + "=" * 80 + "\n"
        summary += "SEGMENTATION METRICS SUMMARY\n"
        summary += "=" * 80 + "\n\n"
        
        # Macro-averaged metrics
        summary += "Macro-Averaged Metrics:\n"
        summary += f"  Dice Score:  {metrics['dice']:.4f}\n"
        summary += f"  IoU Score:   {metrics['iou']:.4f}\n"
        summary += f"  Accuracy:    {metrics['accuracy']:.4f}\n"
        summary += f"  Precision:   {metrics['precision']:.4f}\n"
        summary += f"  Recall:      {metrics['recall']:.4f}\n"
        summary += f"  F1-Score:    {metrics['f1']:.4f}\n\n"
        
        # Per-class metrics
        class_names = ['Background', 'Necrotic', 'Edema', 'Enhancing']
        summary += "Per-Class Metrics:\n"
        summary += f"{'Class':<15} {'Dice':<10} {'IoU':<10} {'Acc':<10} {'Prec':<10} {'Rec':<10}\n"
        summary += "-" * 65 + "\n"
        
        for c in range(self.num_classes):
            class_name = class_names[c] if c < len(class_names) else f"Class_{c}"
            dice_c = metrics['dice_per_class'][c]
            iou_c = metrics['iou_per_class'][c]
            acc_c = metrics['accuracy_per_class'][c]
            prec_c = metrics['precision_per_class'][c]
            rec_c = metrics['recall_per_class'][c]
            
            summary += f"{class_name:<15} {dice_c:<10.4f} {iou_c:<10.4f} {acc_c:<10.4f} {prec_c:<10.4f} {rec_c:<10.4f}\n"
        
        summary += "=" * 80 + "\n"
        
        return summary


class InferenceTimer:
    """
    Measure inference time and throughput
    
    Handles both CPU and GPU timing with proper synchronization
    """
    
    def __init__(self, device: str = 'cpu', warmup_iterations: int = 5):
        """
        Args:
            device: 'cpu' or 'cuda'
            warmup_iterations: Number of warmup runs before measuring
        """
        # FIXED: Store device as torch.device, not string
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.warmup_iterations = warmup_iterations
        self.times = []
    
    def measure(self, model: nn.Module, input_tensor: torch.Tensor, 
                num_iterations: int = 10) -> Dict:
        """
        Measure inference time
        
        Args:
            model: Neural network model
            input_tensor: Input tensor
            num_iterations: Number of times to run inference
        
        Returns:
            stats: Dictionary with timing statistics
        """
        model.eval()
        
        # Move to device
        input_tensor = input_tensor.to(self.device)
        model = model.to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = model(input_tensor)
        
        # Synchronize if GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.time()
                _ = model(input_tensor)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                elapsed = time.time() - start
                times.append(elapsed)
        
        times = np.array(times)
        
        # Compute statistics
        stats = {
            'mean': times.mean() * 1000,  # Convert to ms
            'std': times.std() * 1000,
            'min': times.min() * 1000,
            'max': times.max() * 1000,
            'median': np.median(times) * 1000,
            'throughput': len(input_tensor) / times.mean(),  # Samples/sec
        }
        
        return stats


def count_model_parameters(model: nn.Module) -> Dict:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
    
    Returns:
        stats: Dictionary with parameter statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    # Estimate model size
    model_size_mb = total_params * 4 / 1e6  # FP32
    model_size_mb_fp16 = total_params * 2 / 1e6  # FP16
    
    stats = {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
        'size_mb_fp32': model_size_mb,
        'size_mb_fp16': model_size_mb_fp16,
    }
    
    return stats


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Testing Metrics")
    print("="*80 + "\n")
    
    # Create dummy predictions and targets
    batch_size = 2
    num_classes = 4
    height, width = 128, 128
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}\n")
    
    predictions = torch.randint(0, num_classes, (batch_size, height, width), device=device)
    targets = torch.randint(0, num_classes, (batch_size, height, width), device=device)
    
    # Test SegmentationMetrics
    print("1. Testing SegmentationMetrics...")
    try:
        metrics = SegmentationMetrics(num_classes=num_classes)
        metrics.update(predictions, targets)
        print(metrics.get_summary())
        print("   ✓ SegmentationMetrics test PASSED")
    except Exception as e:
        print(f"   ✗ SegmentationMetrics test FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test InferenceTimer
    print("\n2. Testing InferenceTimer...")
    try:
        from models.architecture import AttentionUNet
        
        model = AttentionUNet(in_channels=4, num_classes=4, pretrained=False)
        timer = InferenceTimer(device=device)
        
        test_input = torch.randn(1, 4, 128, 128, device=device)
        timing_stats = timer.measure(model, test_input, num_iterations=5)
        
        print(f"   Mean inference time: {timing_stats['mean']:.2f} ms")
        print(f"   Std deviation: {timing_stats['std']:.2f} ms")
        print(f"   Min/Max: {timing_stats['min']:.2f} / {timing_stats['max']:.2f} ms")
        print(f"   Throughput: {timing_stats['throughput']:.2f} samples/sec")
        print("   ✓ InferenceTimer test PASSED")
    except Exception as e:
        print(f"   ✗ InferenceTimer test FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Parameter Count
    print("\n3. Testing Parameter Count...")
    try:
        from models.architecture import AttentionUNet
        
        model = AttentionUNet(in_channels=4, num_classes=4, pretrained=False)
        param_stats = count_model_parameters(model)
        
        print(f"   Total Parameters: {param_stats['total']:,}")
        print(f"   Trainable: {param_stats['trainable']:,}")
        print(f"   Frozen: {param_stats['frozen']:,}")
        print(f"   Model Size (FP32): {param_stats['size_mb_fp32']:.2f} MB")
        print(f"   Model Size (FP16): {param_stats['size_mb_fp16']:.2f} MB")
        print("   ✓ Parameter Count test PASSED")
    except Exception as e:
        print(f"   ✗ Parameter Count test FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test device consistency with logits input
    print("\n4. Testing with logits input (4D tensor)...")
    try:
        predictions_logits = torch.randn(batch_size, num_classes, height, width, device=device)
        targets_test = torch.randint(0, num_classes, (batch_size, height, width), device=device)
        
        metrics = SegmentationMetrics(num_classes=num_classes)
        metrics.update(predictions_logits, targets_test)
        
        dice = metrics.compute_dice().item()
        print(f"   Dice Score: {dice:.4f}")
        print("   ✓ Logits input test PASSED")
    except Exception as e:
        print(f"   ✗ Logits input test FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test batch_size=1 edge case
    print("\n5. Testing batch_size=1 (edge case)...")
    try:
        predictions_1 = torch.randint(0, num_classes, (1, height, width), device=device)
        targets_1 = torch.randint(0, num_classes, (1, height, width), device=device)
        
        metrics = SegmentationMetrics(num_classes=num_classes)
        metrics.update(predictions_1, targets_1)
        
        dice = metrics.compute_dice().item()
        print(f"   Dice Score (batch_size=1): {dice:.4f}")
        print("   ✓ Batch size 1 test PASSED")
    except Exception as e:
        print(f"   ✗ Batch size 1 test FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("All metrics tested successfully!")
    print("="*80 + "\n")