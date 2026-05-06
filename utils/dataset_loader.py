# ============ UTILS/DATASET_LOADER.PY - COMPLETE FIXED CODE ============

"""
Dataset Loader for Brain MRI Segmentation

Loads preprocessed NumPy arrays from processed_data/ directory
Handles data augmentation and batching
Critical fixes:
1. All imports use modular paths
2. Fixed seed for reproducible splits (no split changes per run)
3. Handles edge cases (empty slices, missing files)
4. Patient-level splits prevent data leakage
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


class BrainMRIDataset(Dataset):
    """
    PyTorch Dataset for brain MRI segmentation
    
    Loads preprocessed 2D slices:
    - Images: 128x128x4 (4 MRI modalities)
    - Masks: 128x128 (4-class segmentation)
    
    Handles train/val/test splits by patient to prevent data leakage
    """
    
    def __init__(self, data_dir: str, metadata_csv: str, split: str = 'train',
                 val_split: float = 0.2, test_split: float = 0.1,
                 transforms=None, seed: int = 42):
        """
        Args:
            data_dir: Path to processed_data/ directory
            metadata_csv: Path to metadata.csv file
            split: 'train', 'val', or 'test'
            val_split: Validation split ratio (0.2 = 20%)
            test_split: Test split ratio (0.1 = 10%)
            transforms: Optional augmentation transforms
            seed: Random seed for reproducible splits
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / 'images'
        self.masks_dir = self.data_dir / 'masks'
        self.split = split
        self.transforms = transforms
        
        # Verify directories exist
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")
        
        # Load metadata
        if not Path(metadata_csv).exists():
            raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
        
        self.metadata = pd.read_csv(metadata_csv)
        logger.info(f"Loaded metadata with {len(self.metadata)} slices")
        
        # Get unique patients
        unique_patients = self.metadata['patient_id'].unique()
        num_patients = len(unique_patients)
        logger.info(f"Found {num_patients} unique patients")
        
        # Set random seed for reproducible splits (FIXED: use fixed seed)
        np.random.seed(seed)
        
        # Shuffle patients
        indices = np.arange(num_patients)
        np.random.shuffle(indices)
        
        # Calculate split points
        test_size = int(num_patients * test_split)
        val_size = int(num_patients * val_split)
        train_size = num_patients - test_size - val_size
        
        # Get patient groups
        train_patients = unique_patients[indices[:train_size]]
        val_patients = unique_patients[indices[train_size:train_size + val_size]]
        test_patients = unique_patients[indices[train_size + val_size:]]
        
        logger.info(f"Train patients: {len(train_patients)}, "
                   f"Val patients: {len(val_patients)}, "
                   f"Test patients: {len(test_patients)}")
        
        # Filter metadata by split
        if split == 'train':
            self.metadata = self.metadata[self.metadata['patient_id'].isin(train_patients)]
        elif split == 'val':
            self.metadata = self.metadata[self.metadata['patient_id'].isin(val_patients)]
        elif split == 'test':
            self.metadata = self.metadata[self.metadata['patient_id'].isin(test_patients)]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Reset index
        self.metadata = self.metadata.reset_index(drop=True)
        
        logger.info(f"Loaded {len(self.metadata)} slices for {split} split")
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get single sample
        
        Returns:
            image: [4, 128, 128] - 4-channel MRI
            mask: [128, 128] - class indices (0-3)
        """
        row = self.metadata.iloc[idx]
        
        # Load image and mask
        image_path = self.images_dir / row['image_name']
        mask_path = self.masks_dir / row['mask_name']
        
        # Verify files exist
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        try:
            image = np.load(image_path).astype(np.float32)  # [128, 128, 4]
            mask = np.load(mask_path).astype(np.int64)  # [128, 128], FIXED: np.int64 not np.long
        except Exception as e:
            logger.error(f"Error loading {image_path} or {mask_path}: {e}")
            raise
        
        # Rearrange to [C, H, W] for PyTorch
        image = np.transpose(image, (2, 0, 1))  # [4, 128, 128]
        
        # Apply transforms if provided
        if self.transforms:
            image, mask = self.transforms(image, mask)
        
        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()
        
        return image, mask


class BrainMRIDataModule:
    """
    Convenient wrapper for creating train/val/test dataloaders
    Ensures consistent splits across multiple runs (fixed seed)
    """
    
    def __init__(self, data_dir: str, metadata_csv: str, batch_size: int = 16,
                 num_workers: int = 4, val_split: float = 0.2, test_split: float = 0.1,
                 transforms_train=None, transforms_val=None, seed: int = 42):
        """
        Args:
            data_dir: Path to processed_data/
            metadata_csv: Path to metadata.csv
            batch_size: Batch size for dataloaders
            num_workers: Number of data loading workers
            val_split: Validation split ratio
            test_split: Test split ratio
            transforms_train: Augmentation for training
            transforms_val: Augmentation for validation/test
            seed: Random seed (FIXED: ensures reproducible splits)
        """
        self.data_dir = data_dir
        self.metadata_csv = metadata_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.transforms_train = transforms_train
        self.transforms_val = transforms_val
        self.seed = seed
        
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
    
    def setup(self):
        """Create datasets with fixed seed for reproducibility"""
        # FIXED: Use same seed for all splits to ensure consistency
        self._train_dataset = BrainMRIDataset(
            data_dir=self.data_dir,
            metadata_csv=self.metadata_csv,
            split='train',
            val_split=self.val_split,
            test_split=self.test_split,
            transforms=self.transforms_train,
            seed=self.seed  # FIXED: Use same seed
        )
        
        self._val_dataset = BrainMRIDataset(
            data_dir=self.data_dir,
            metadata_csv=self.metadata_csv,
            split='val',
            val_split=self.val_split,
            test_split=self.test_split,
            transforms=self.transforms_val,
            seed=self.seed  # FIXED: Use same seed
        )
        
        self._test_dataset = BrainMRIDataset(
            data_dir=self.data_dir,
            metadata_csv=self.metadata_csv,
            split='test',
            val_split=self.val_split,
            test_split=self.test_split,
            transforms=self.transforms_val,
            seed=self.seed  # FIXED: Use same seed
        )

        from torch.utils.data import Subset
        import random

        random.seed(self.seed)
        train_size = min(25000, len(self._train_dataset))
        train_indices = random.sample(range(len(self._train_dataset)), train_size)
        self._train_dataset = Subset(self._train_dataset, train_indices)
        val_size = min(5000, len(self._val_dataset))
        val_indices = random.sample(range(len(self._val_dataset)), val_size)
        self._val_dataset = Subset(self._val_dataset, val_indices)
        print(f"✅ Using {train_size} training samples and {val_size} validation samples")
       
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader"""
        if self._train_dataset is None:
            self.setup()
        
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader"""
        if self._val_dataset is None:
            self.setup()
        
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader"""
        if self._test_dataset is None:
            self.setup()
        
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Testing Dataset Loader")
    print("="*80 + "\n")
    
    # Example usage (requires preprocessed data)
    data_dir = "processed_data"
    metadata_csv = "processed_data/metadata.csv"
    
    if os.path.exists(metadata_csv):
        print(f"Loading data from {data_dir}...")
        
        try:
            # Create data module with fixed seed
            dm = BrainMRIDataModule(
                data_dir=data_dir,
                metadata_csv=metadata_csv,
                batch_size=4,
                num_workers=2,
                seed=42  # FIXED: reproducible splits
            )
            dm.setup()
            
            # Get dataloaders
            train_loader = dm.train_dataloader()
            val_loader = dm.val_dataloader()
            test_loader = dm.test_dataloader()
            
            print(f"✓ Train dataset size: {len(dm._train_dataset)}")
            print(f"✓ Val dataset size: {len(dm._val_dataset)}")
            print(f"✓ Test dataset size: {len(dm._test_dataset)}")
            
            # Test loading a batch
            print("\nLoading a batch from training set...")
            images, masks = next(iter(train_loader))
            print(f"✓ Batch shape: {images.shape}")
            print(f"✓ Mask shape: {masks.shape}")
            print(f"✓ Image value range: [{images.min():.2f}, {images.max():.2f}]")
            print(f"✓ Unique classes in batch: {torch.unique(masks).tolist()}")
            
            print("\n" + "="*80)
            print("Dataset loader test PASSED!")
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠ Warning: {metadata_csv} not found")
        print("Please run preprocessing first!")
        print("\nExample preprocessing command:")
        print("  python scripts/preprocess_local.py \\")
        print("    --input_dir data/raw \\")
        print("    --output_dir processed_data")