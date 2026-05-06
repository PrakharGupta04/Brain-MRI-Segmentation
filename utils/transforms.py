# ============ UTILS/TRANSFORMS.PY - COMPLETE FIXED CODE (KEY FIX) ============

"""
Data Augmentation and Helper Functions

Transforms for training:
- RandomHorizontalFlip
- RandomVerticalFlip
- RandomRotation
- RandomAffine
- GaussianNoise
- Compose

CRITICAL FIX: np.long → np.int64 (np.long is deprecated)
"""

import numpy as np
import cv2
import torch
from scipy import ndimage
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


class Compose:
    """Compose multiple transforms"""
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask


class RandomHorizontalFlip:
    """Random horizontal flip"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() < self.p:
            image = np.flip(image, axis=-1).copy()  # Flip W axis [C, H, W]
            mask = np.flip(mask, axis=-1).copy()    # Flip W axis [H, W]
        return image, mask


class RandomVerticalFlip:
    """Random vertical flip"""
    
    def __init__(self, p: float = 0.3):
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() < self.p:
            image = np.flip(image, axis=-2).copy()  # Flip H axis [C, H, W]
            mask = np.flip(mask, axis=-2).copy()    # Flip H axis [H, W]
        return image, mask


class RandomRotation:
    """Random rotation"""
    
    def __init__(self, degrees: float = 10):
        self.degrees = degrees
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        angle = np.random.uniform(-self.degrees, self.degrees)
        
        # Image: [C, H, W] → rotate each channel
        image_rotated = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[0]):
            image_rotated[c] = ndimage.rotate(image[c], angle, reshape=False, order=1)
        
        # Mask: [H, W] → rotate (FIXED: use np.int64)
        mask_rotated = ndimage.rotate(mask.astype(np.float32), angle, reshape=False, order=0)
        mask_rotated = mask_rotated.astype(np.int64)  # FIXED: was np.long
        
        return image_rotated, mask_rotated


class RandomAffine:
    """Random affine transformation (zoom)"""
    
    def __init__(self, scale: Tuple[float, float] = (0.9, 1.1)):
        self.scale = scale
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Random scale
        scale_factor = np.random.uniform(self.scale[0], self.scale[1])
        
        # Get shape [C, H, W]
        h, w = image.shape[-2:]
        
        # Create transformation matrix (zoom from center)
        center = np.array([w / 2, h / 2])
        M = cv2.getRotationMatrix2D(tuple(center), 0, scale_factor)
        
        # Apply to each channel
        image_transformed = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[0]):
            image_transformed[c] = cv2.warpAffine(
                image[c].astype(np.float32),
                M, (w, h),
                borderMode=cv2.BORDER_REFLECT
            )
        
        # Apply to mask (FIXED: use np.int64)
        mask_transformed = cv2.warpAffine(
            mask.astype(np.float32),
            M, (w, h),
            borderMode=cv2.BORDER_REFLECT,
            flags=cv2.INTER_NEAREST
        )
        mask_transformed = mask_transformed.astype(np.int64)  # FIXED: was np.long
        
        return image_transformed, mask_transformed


class GaussianNoise:
    """Add Gaussian noise"""
    
    def __init__(self, mean: float = 0.0, std: float = 0.01):
        self.mean = mean
        self.std = std
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        noise = np.random.normal(self.mean, self.std, image.shape)
        image = image + noise
        return image, mask


class RandomGaussianBlur:
    """Random Gaussian blur"""
    
    def __init__(self, kernel_size: int = 3, sigma: Tuple[float, float] = (0.1, 2.0), p: float = 0.3):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() < self.p:
            sigma = np.random.uniform(self.sigma[0], self.sigma[1])
            
            # Apply to each channel
            image_blurred = np.zeros_like(image)
            for c in range(image.shape[0]):
                image_blurred[c] = cv2.GaussianBlur(
                    image[c],
                    (self.kernel_size, self.kernel_size),
                    sigma
                )
            
            return image_blurred, mask
        
        return image, mask


def get_augmentation_pipeline(stage: str = 'train'):
    """
    Get standard augmentation pipeline
    
    Args:
        stage: 'train' or 'val'
    
    Returns:
        transforms: Compose object with augmentation pipeline
    """
    if stage == 'train':
        transforms = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.3),
            RandomRotation(degrees=10),
            RandomAffine(scale=(0.9, 1.1)),
            GaussianNoise(mean=0.0, std=0.01),
            RandomGaussianBlur(kernel_size=3, sigma=(0.1, 1.0), p=0.2),
        ])
    else:
        # No augmentation for validation/test
        transforms = Compose([])
    
    return transforms


# ============ Helper Functions ============

def normalize_mri(volume: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize MRI volume
    
    Args:
        volume: Numpy array [H, W, D] or [H, W]
        method: 'zscore' or 'minmax'
    
    Returns:
        normalized: Normalized volume
    """
    if method == 'zscore':
        # Z-score normalization
        mean = volume.mean()
        std = volume.std()
        normalized = (volume - mean) / (std + 1e-6)
    elif method == 'minmax':
        # Min-max normalization
        v_min = volume.min()
        v_max = volume.max()
        normalized = (volume - v_min) / (v_max - v_min + 1e-6)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return normalized


def clip_outliers(volume: np.ndarray, percentile_low: float = 2, percentile_high: float = 98) -> np.ndarray:
    """
    Clip outlier intensities
    
    Args:
        volume: Numpy array
        percentile_low: Low percentile
        percentile_high: High percentile
    
    Returns:
        clipped: Clipped volume
    """
    v_low = np.percentile(volume, percentile_low)
    v_high = np.percentile(volume, percentile_high)
    clipped = np.clip(volume, v_low, v_high)
    return clipped


def resize_2d(image: np.ndarray, size: int = 128, order: int = 1) -> np.ndarray:
    """
    Resize 2D image
    
    Args:
        image: Numpy array [H, W] or [C, H, W]
        size: Target size (square)
        order: Interpolation order (1=bilinear)
    
    Returns:
        resized: Resized image
    """
    if image.ndim == 2:
        resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    elif image.ndim == 3:
        # Multi-channel
        resized = np.zeros((image.shape[0], size, size), dtype=image.dtype)
        for c in range(image.shape[0]):
            resized[c] = cv2.resize(image[c], (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    return resized


def stack_modalities(t1n: np.ndarray, t1c: np.ndarray, t2w: np.ndarray, t2f: np.ndarray) -> np.ndarray:
    """
    Stack 4 modalities into single array
    
    Args:
        t1n, t1c, t2w, t2f: Individual modality volumes [H, W, D]
    
    Returns:
        stacked: [H, W, D, 4]
    """
    stacked = np.stack([t1n, t1c, t2w, t2f], axis=-1)
    return stacked


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Testing Augmentation Transforms")
    print("="*80 + "\n")
    
    # Create dummy data
    image = np.random.randn(4, 128, 128).astype(np.float32)
    mask = np.random.randint(0, 4, (128, 128)).astype(np.int64)
    
    print(f"Input shapes: image={image.shape}, mask={mask.shape}")
    print(f"Input dtypes: image={image.dtype}, mask={mask.dtype}")
    
    # Test pipeline
    print("\nTesting training augmentation pipeline...")
    try:
        pipeline = get_augmentation_pipeline(stage='train')
        image_aug, mask_aug = pipeline(image.copy(), mask.copy())
        
        print(f"✓ Output shapes: image={image_aug.shape}, mask={mask_aug.shape}")
        print(f"✓ Output dtypes: image={image_aug.dtype}, mask={mask_aug.dtype}")
        print(f"✓ Mask dtype is np.int64 (not deprecated np.long): {mask_aug.dtype == np.int64}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Transforms test completed!")
    print("="*80 + "\n")