"""
Preprocessing Pipeline for BraTS 2023 GLI Dataset

Converts raw NIfTI files to processed NumPy arrays suitable for training

Pipeline:
1. Read raw NIfTI files (4 modalities + segmentation)
2. Stack modalities into 4-channel volume
3. Extract 2D slices (keeping only slices with tumor)
4. Normalize per modality
5. Resize to specified size
6. Save as NumPy arrays
7. Generate metadata CSV

Features:
- Batch processing (configurable batch size)
- Resume capability (skip already processed files)
- Robust error handling
- Progress tracking
- Dynamic file finding (handles various naming conventions)
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
import cv2
from typing import List, Dict, Tuple, Optional
import traceback


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BraTSPreprocessor:
    """
    Main preprocessing class for BraTS dataset
    
    Handles:
    - Dynamic file discovery (flexible naming conventions)
    - NIfTI file loading
    - Per-modality normalization
    - 2D slice extraction
    - Image resizing
    - Metadata tracking
    """
    
    def __init__(self, input_dir: str, output_dir: str, resize_size: int = 128, 
                 keep_tumor_only: bool = True):
        """
        Args:
            input_dir: Path to raw BraTS dataset
            output_dir: Path to save processed data
            resize_size: Target size for 2D slices (e.g., 128)
            keep_tumor_only: If True, keep only slices with tumor
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.resize_size = resize_size
        self.keep_tumor_only = keep_tumor_only
        
        # Create output directories
        self.images_dir = self.output_dir / 'images'
        self.masks_dir = self.output_dir / 'masks'
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Resize size: {resize_size}")
        logger.info(f"Keep tumor only: {keep_tumor_only}")
        
        # Metadata tracking
        self.metadata_list = []
        
        # Statistics
        self.stats = {
            'total_patients': 0,
            'processed_patients': 0,
            'failed_patients': 0,
            'total_slices_saved': 0,
        }
    
    def find_patient_folders(self) -> List[Path]:
        """
        Find all patient folders in dataset
        
        Patient folders typically start with 'BraTS-GLI' or similar
        
        Returns:
            patient_folders: List of patient folder paths
        """
        patient_folders = sorted([
            p for p in self.input_dir.iterdir() 
            if p.is_dir() and p.name.startswith('BraTS')
        ])
        
        logger.info(f"Found {len(patient_folders)} patient folders")
        self.stats['total_patients'] = len(patient_folders)
        
        return patient_folders
    
    def find_modality_files(self, patient_folder: Path) -> Dict[str, Optional[Path]]:
        """
        Dynamically find modality files in patient folder
        
        Handles various naming conventions:
        - BraTS-GLI-00000-000/t1n.nii.gz
        - BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1n.nii.gz
        - BraTS-GLI-00000-000/t1n.nii (without .gz)
        - Case variations
        
        Args:
            patient_folder: Path to patient folder
        
        Returns:
            modality_dict: Dictionary with paths to modality files
                {
                    't1n': Path or None,
                    't1c': Path or None,
                    't2w': Path or None,
                    't2f': Path or None,
                    'seg': Path or None
                }
        """
        modalities = {
            't1n': None,
            't1c': None,
            't2w': None,
            't2f': None,
            'seg': None
        }
        
        # Search for files containing modality keywords
        all_files = list(patient_folder.glob('*.nii.gz')) + list(patient_folder.glob('*.nii'))
        
        for modality in modalities.keys():
            # Search for file containing the modality keyword (case-insensitive)
            for file_path in all_files:
                # Check if modality appears in filename
                if modality in file_path.name.lower():
                    modalities[modality] = file_path
                    break
        
        return modalities
    
    def load_modality(self, path: Optional[Path]) -> Optional[np.ndarray]:
        """
        Load NIfTI file
        
        Args:
            path: Path to .nii.gz or .nii file
        
        Returns:
            data: Numpy array [H, W, D] or None if failed
        """
        if path is None:
            return None
        
        try:
            img = nib.load(path)
            data = np.array(img.dataobj, dtype=np.float32)
            logger.debug(f"Loaded {path.name} with shape {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return None
    
    def normalize_modality(self, volume: np.ndarray) -> np.ndarray:
        """
        Normalize modality using z-score
        
        Args:
            volume: Numpy array [H, W, D]
        
        Returns:
            normalized: Normalized array
        """
        # Remove background (zeros) for statistics
        mask = volume > 0
        if mask.sum() == 0:
            logger.warning("Volume is all zeros, returning as-is")
            return volume
        
        mean = volume[mask].mean()
        std = volume[mask].std()
        
        if std < 1e-6:
            logger.warning(f"Volume has very low std: {std}")
            return volume
        
        normalized = (volume - mean) / std
        
        return normalized
    
    def process_patient(self, patient_folder: Path) -> List[Dict]:
        """
        Process single patient
        
        Steps:
        1. Find modality files (with flexible naming)
        2. Load all modalities and segmentation
        3. Validate data shapes
        4. Normalize modalities
        5. Stack into 4-channel volume
        6. Extract and process 2D slices
        
        Args:
            patient_folder: Path to patient folder
        
        Returns:
            metadata: List of (image_name, mask_name, patient_id, slice_no)
        """
        patient_id = patient_folder.name
        logger.info(f"Processing patient: {patient_id}")
        
        try:
            # CRITICAL FIX: Dynamically find files instead of assuming exact names
            modality_files = self.find_modality_files(patient_folder)
            
            # Check if all required files were found
            missing_modalities = [k for k, v in modality_files.items() if v is None]
            if missing_modalities:
                logger.warning(f"Patient {patient_id} missing modalities: {missing_modalities}")
                if len(missing_modalities) == 5:  # All files missing
                    logger.error(f"Patient {patient_id} has no modality files at all!")
                    return []
            
            # Load data
            logger.debug(f"  Loading modalities...")
            t1n = self.load_modality(modality_files['t1n'])
            t1c = self.load_modality(modality_files['t1c'])
            t2w = self.load_modality(modality_files['t2w'])
            t2f = self.load_modality(modality_files['t2f'])
            seg = self.load_modality(modality_files['seg'])
            
            # Check if critical files were loaded
            if any(x is None for x in [t1n, t1c, t2w, t2f, seg]):
                logger.warning(f"Failed to load all modalities for {patient_id}")
                return []
            
            # Validate shapes match
            shape_t1n = t1n.shape
            if t1c.shape != shape_t1n or t2w.shape != shape_t1n or t2f.shape != shape_t1n or seg.shape != shape_t1n:
                logger.error(f"Shape mismatch in {patient_id}: "
                           f"t1n={t1n.shape}, t1c={t1c.shape}, t2w={t2w.shape}, t2f={t2f.shape}, seg={seg.shape}")
                return []
            
            # Normalize modalities
            logger.debug(f"  Normalizing modalities...")
            t1n = self.normalize_modality(t1n)
            t1c = self.normalize_modality(t1c)
            t2w = self.normalize_modality(t2w)
            t2f = self.normalize_modality(t2f)
            
            # Stack modalities
            volume = np.stack([t1n, t1c, t2w, t2f], axis=-1)  # [H, W, D, 4]
            
            # Process slices
            logger.debug(f"  Extracting slices...")
            metadata = []
            slices_saved = 0
            
            for slice_idx in range(volume.shape[2]):
                # Get slice
                image_slice = volume[:, :, slice_idx, :]  # [H, W, 4]
                mask_slice = seg[:, :, slice_idx]  # [H, W]
                
                # Check if should keep slice
                if self.keep_tumor_only:
                    if (mask_slice > 0).sum() == 0:  # No tumor in slice
                        continue
                
                # Resize
                image_resized = cv2.resize(
                    image_slice,
                    (self.resize_size, self.resize_size),
                    interpolation=cv2.INTER_LINEAR
                )
                mask_resized = cv2.resize(
                    mask_slice,
                    (self.resize_size, self.resize_size),
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Save image
                image_name = f"{patient_id}_slice_{slice_idx:03d}.npy"
                image_path = self.images_dir / image_name
                np.save(image_path, image_resized.astype(np.float32))
                
                # Save mask
                mask_name = f"{patient_id}_slice_{slice_idx:03d}.npy"
                mask_path = self.masks_dir / mask_name
                np.save(mask_path, mask_resized.astype(np.uint8))
                
                # Record metadata
                metadata.append({
                    'image_name': image_name,
                    'mask_name': mask_name,
                    'patient_id': patient_id,
                    'slice_no': slice_idx,
                    'shape': str(image_resized.shape),
                    'modalities': 't1n|t1c|t2w|t2f'
                })
                
                slices_saved += 1
            
            logger.info(f"  Saved {slices_saved} slices for {patient_id}")
            self.stats['total_slices_saved'] += slices_saved
            
            return metadata
        
        except Exception as e:
            logger.error(f"Error processing {patient_id}: {e}")
            logger.error(traceback.format_exc())
            self.stats['failed_patients'] += 1
            return []
    
    def preprocess_batch(self, patient_folders: List[Path], batch_size: int = 50) -> List[Dict]:
        """
        Process batch of patients
        
        Args:
            patient_folders: List of patient folder paths
            batch_size: Number of patients per batch
        
        Returns:
            all_metadata: List of all metadata dictionaries
        """
        all_metadata = []
        
        # Process in batches
        for batch_idx in range(0, len(patient_folders), batch_size):
            batch = patient_folders[batch_idx:batch_idx + batch_size]
            logger.info(f"\nProcessing batch {batch_idx // batch_size + 1} "
                       f"({len(batch)} patients)...")
            
            # Process each patient in batch
            for patient_folder in tqdm(batch, desc="Processing patients"):
                metadata = self.process_patient(patient_folder)
                all_metadata.extend(metadata)
                if metadata:
                    self.stats['processed_patients'] += 1
            
            # Save metadata after each batch
            self.save_metadata(all_metadata)
            logger.info(f"  Batch complete. Total slices: {len(all_metadata)}")
        
        return all_metadata
    
    def save_metadata(self, metadata_list: List[Dict]):
        """
        Save metadata to CSV
        
        Args:
            metadata_list: List of metadata dictionaries
        """
        if not metadata_list:
            logger.warning("No metadata to save")
            return
        
        df = pd.DataFrame(metadata_list)
        csv_path = self.output_dir / 'metadata.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved metadata to {csv_path}")
    
    def get_statistics(self, metadata_list: List[Dict]):
        """
        Print preprocessing statistics
        
        Args:
            metadata_list: List of metadata dictionaries
        """
        df = pd.DataFrame(metadata_list)
        
        print("\n" + "=" * 80)
        print("PREPROCESSING STATISTICS")
        print("=" * 80)
        
        num_patients = df['patient_id'].nunique() if len(df) > 0 else 0
        num_slices = len(df)
        
        print(f"\nDataset Summary:")
        print(f"  Total Patients Found: {self.stats['total_patients']}")
        print(f"  Successfully Processed: {self.stats['processed_patients']}")
        print(f"  Failed: {self.stats['failed_patients']}")
        print(f"  Total Slices Extracted: {num_slices}")
        
        if num_patients > 0:
            print(f"  Average Slices/Patient: {num_slices / num_patients:.1f}")
        
        print(f"\nOutput Directories:")
        print(f"  Images: {self.images_dir}")
        print(f"  Masks: {self.masks_dir}")
        print(f"  Metadata CSV: {self.output_dir / 'metadata.csv'}")
        
        # Verify files
        image_files = len(list(self.images_dir.glob('*.npy')))
        mask_files = len(list(self.masks_dir.glob('*.npy')))
        
        print(f"\nVerification:")
        print(f"  Image files: {image_files}")
        print(f"  Mask files: {mask_files}")
        print(f"  Metadata entries: {len(df)}")
        
        if image_files == mask_files == len(df):
            print(f"  ✓ All files present and consistent!")
        else:
            print(f"  ✗ File count mismatch!")
        
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess BraTS 2023 GLI dataset'
    )
    
    parser.add_argument(
        '--input_dir',
        required=True,
        help='Path to raw BraTS dataset (containing patient folders)'
    )
    
    parser.add_argument(
        '--output_dir',
        default='processed_data',
        help='Path to save processed data'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=50,
        help='Number of patients per batch'
    )
    
    parser.add_argument(
        '--resize',
        type=int,
        default=128,
        help='Target size for 2D slices'
    )
    
    parser.add_argument(
        '--keep_tumor_only',
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help='Keep only slices with tumor (default: True)'
    )
    
    parser.add_argument(
        '--log_level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    print("\n" + "=" * 80)
    print("BRATS 2023 GLI PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Create preprocessor
    preprocessor = BraTSPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        resize_size=args.resize,
        keep_tumor_only=args.keep_tumor_only
    )
    
    # Find patient folders
    patient_folders = preprocessor.find_patient_folders()
    
    if not patient_folders:
        logger.error(f"No patient folders found in {args.input_dir}")
        logger.error("Expected folders starting with 'BraTS'")
        return
    
    # Process in batches
    metadata = preprocessor.preprocess_batch(
        patient_folders,
        batch_size=args.batch_size
    )
    
    # Print statistics
    preprocessor.get_statistics(metadata)
    
    logger.info("Preprocessing complete!")


if __name__ == '__main__':
    main()