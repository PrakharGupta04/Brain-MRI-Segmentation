"""
Helper functions for file I/O, NIfTI handling, and data processing
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


def load_nifti(filepath, dtype=np.float32):
    """
    Load NIfTI file
    
    Args:
        filepath: Path to .nii.gz file
        dtype: Output data type
    
    Returns:
        data: Numpy array
        affine: Affine transformation matrix
    """
    try:
        img = nib.load(filepath)
        data = np.array(img.dataobj, dtype=dtype)
        affine = img.affine
        logger.info(f"Loaded {filepath} with shape {data.shape}")
        return data, affine
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None, None


def save_nifti(data, filepath, affine=None):
    """
    Save data as NIfTI file
    
    Args:
        data: Numpy array
        filepath: Output path
        affine: Affine transformation matrix (optional)
    """
    try:
        if affine is None:
            affine = np.eye(4)
        
        img = nib.Nifti1Image(data, affine)
        nib.save(img, filepath)
        logger.info(f"Saved {filepath}")
    except Exception as e:
        logger.error(f"Error saving {filepath}: {e}")


def normalize_volume(volume, method='zscore', mask=None):
    """
    Normalize 3D volume
    
    Args:
        volume: 3D numpy array [H, W, D]
        method: 'zscore' or 'minmax'
        mask: Optional binary mask for computing statistics
    
    Returns:
        normalized: Normalized volume
    """
    if mask is None:
        mask = volume > 0
    
    if method == 'zscore':
        mean = volume[mask].mean()
        std = volume[mask].std()
        normalized = (volume - mean) / (std + 1e-6)
    elif method == 'minmax':
        v_min = volume[mask].min()
        v_max = volume[mask].max()
        normalized = (volume - v_min) / (v_max - v_min + 1e-6)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return normalized


def crop_to_content(volume, margin=10):
    """
    Crop volume to remove background
    
    Args:
        volume: 3D numpy array
        margin: Margin to keep around content
    
    Returns:
        cropped: Cropped volume
        bbox: Bounding box (z_min, z_max, y_min, y_max, x_min, x_max)
    """
    mask = volume > 0
    
    # Find bounding box
    z_indices = np.any(mask, axis=(1, 2))
    y_indices = np.any(mask, axis=(0, 2))
    x_indices = np.any(mask, axis=(0, 1))
    
    z_min, z_max = np.where(z_indices)[0][[0, -1]]
    y_min, y_max = np.where(y_indices)[0][[0, -1]]
    x_min, x_max = np.where(x_indices)[0][[0, -1]]
    
    # Add margin
    z_min = max(0, z_min - margin)
    z_max = min(volume.shape[0], z_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(volume.shape[1], y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(volume.shape[2], x_max + margin)
    
    cropped = volume[z_min:z_max, y_min:y_max, x_min:x_max]
    bbox = (z_min, z_max, y_min, y_max, x_min, x_max)
    
    return cropped, bbox


def pad_to_size(volume, target_size, pad_value=0):
    """
    Pad volume to target size
    
    Args:
        volume: 3D numpy array [D, H, W]
        target_size: Tuple (D, H, W)
        pad_value: Value to pad with
    
    Returns:
        padded: Padded volume
    """
    current_size = volume.shape
    
    if current_size == target_size:
        return volume
    
    # Calculate padding
    pad_total = [target_size[i] - current_size[i] for i in range(3)]
    pad_before = [p // 2 for p in pad_total]
    pad_after = [p - pb for p, pb in zip(pad_total, pad_before)]
    
    # Create padding tuple for numpy
    padding = [(pad_before[i], pad_after[i]) for i in range(3)]
    
    padded = np.pad(volume, padding, mode='constant', constant_values=pad_value)
    
    return padded


def resample_volume(volume, scale_factor):
    """
    Resample 3D volume
    
    Args:
        volume: 3D numpy array
        scale_factor: Resampling factor or tuple of factors
    
    Returns:
        resampled: Resampled volume
    """
    from scipy.ndimage import zoom
    
    if isinstance(scale_factor, (int, float)):
        scale_factor = (scale_factor, scale_factor, scale_factor)
    
    resampled = zoom(volume, scale_factor, order=1)
    
    return resampled


def dice_coefficient(pred, target, smooth=1.0):
    """
    Compute Dice coefficient
    
    Args:
        pred: Predicted segmentation [H, W, D]
        target: Target segmentation [H, W, D]
        smooth: Smoothing constant
    
    Returns:
        dice: Dice coefficient [0, 1]
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice


def iou_coefficient(pred, target, smooth=1.0):
    """
    Compute Intersection over Union
    
    Args:
        pred: Predicted segmentation
        target: Target segmentation
        smooth: Smoothing constant
    
    Returns:
        iou: IoU coefficient [0, 1]
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = (pred_flat * target_flat).sum()
    union = (pred_flat.sum() + target_flat.sum() - intersection)
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


def get_largest_connected_component(binary_mask):
    """
    Get largest connected component from binary mask
    
    Args:
        binary_mask: Binary 3D mask [H, W, D]
    
    Returns:
        largest_component: Largest connected component
    """
    from scipy.ndimage import label, find_objects
    
    # Label connected components
    labeled_array, num_features = label(binary_mask)
    
    if num_features == 0:
        return binary_mask
    
    # Find sizes
    component_sizes = np.bincount(labeled_array.ravel())
    
    # Get largest component (excluding background 0)
    largest_label = np.argmax(component_sizes[1:]) + 1
    largest_component = (labeled_array == largest_label).astype(np.uint8)
    
    return largest_component


def compute_surface_distance(pred, target):
    """
    Compute surface distance between two segmentations (Hausdorff distance)
    
    Args:
        pred: Predicted segmentation
        target: Target segmentation
    
    Returns:
        hausdorff_dist: Hausdorff distance
    """
    from scipy.spatial.distance import cdist
    from scipy.ndimage import binary_erosion
    
    # Get surface (edges)
    pred_surface = binary_erosion(pred) != pred
    target_surface = binary_erosion(target) != target
    
    # Get coordinates
    pred_coords = np.array(np.where(pred_surface)).T
    target_coords = np.array(np.where(target_surface)).T
    
    if len(pred_coords) == 0 or len(target_coords) == 0:
        return 0.0
    
    # Compute distances
    distances = cdist(pred_coords, target_coords, metric='euclidean')
    
    # Hausdorff distance
    hausdorff = max(distances.min(axis=1).max(), distances.min(axis=0).max())
    
    return hausdorff


class FileManager:
    """Utility class for file management"""
    
    @staticmethod
    def ensure_dir(dirpath):
        """Ensure directory exists"""
        Path(dirpath).mkdir(parents=True, exist_ok=True)
        return Path(dirpath)
    
    @staticmethod
    def get_patient_folders(dataset_dir):
        """Get list of patient folders"""
        dataset_path = Path(dataset_dir)
        patient_folders = sorted([
            p for p in dataset_path.iterdir()
            if p.is_dir() and p.name.startswith('BraTS')
        ])
        return patient_folders
    
    @staticmethod
    def get_modality_files(patient_folder):
        """Get modality files from patient folder"""
        patient_path = Path(patient_folder)
        
        modalities = {
            't1n': patient_path / 't1n.nii.gz',
            't1c': patient_path / 't1c.nii.gz',
            't2w': patient_path / 't2w.nii.gz',
            't2f': patient_path / 't2f.nii.gz',
            'seg': patient_path / 'seg.nii.gz',
        }
        
        return modalities
    
    @staticmethod
    def check_files_exist(modalities_dict):
        """Check if all modality files exist"""
        missing = []
        for name, path in modalities_dict.items():
            if not Path(path).exists():
                missing.append(name)
        
        return len(missing) == 0, missing


if __name__ == "__main__":
    print("Testing helper functions...")
    
    # Test normalization
    volume = np.random.randn(32, 32, 32) * 100 + 50
    normalized = normalize_volume(volume, method='zscore')
    print(f"✓ Normalization: mean={normalized.mean():.3f}, std={normalized.std():.3f}")
    
    # Test cropping
    volume_large = np.zeros((64, 64, 64))
    volume_large[20:40, 20:40, 20:40] = 1
    cropped, bbox = crop_to_content(volume_large)
    print(f"✓ Cropping: {volume_large.shape} → {cropped.shape}")
    
    # Test padding
    volume_small = np.ones((32, 32, 32))
    padded = pad_to_size(volume_small, (64, 64, 64))
    print(f"✓ Padding: {volume_small.shape} → {padded.shape}")
    
    # Test Dice
    pred = np.ones((32, 32, 32))
    target = np.ones((32, 32, 32))
    dice = dice_coefficient(pred, target)
    print(f"✓ Dice coefficient: {dice:.3f} (should be 1.0)")
    
    # Test FileManager
    fm = FileManager()
    print(f"✓ FileManager initialized")
    
    print("\nAll helper functions tested successfully!")
