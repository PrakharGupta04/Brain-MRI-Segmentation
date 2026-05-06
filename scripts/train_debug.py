import sys
from pathlib import Path
import torch
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
import os

# Fix imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.architecture import AttentionUNet
from models.losses import CombinedLoss
from utils.dataset_loader import BrainMRIDataModule
from utils.transforms import get_augmentation_pipeline


def main():
    # ================= CONFIG =================
    DATA_DIR = "E:/Brain_MRI_DL/processed_data"
    BATCH_SIZE = 4
    TRAIN_SAMPLES = 2000
    VAL_SAMPLES = 500
    EPOCHS = 2

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================= LOAD DATA =================
    print("Loading data...")

    dm = BrainMRIDataModule(
        data_dir=DATA_DIR,
        metadata_csv=f"{DATA_DIR}/metadata.csv",
        batch_size=BATCH_SIZE,
        num_workers=2,   # safe now
        val_split=0.2,
        test_split=0.1,
        transforms_train=get_augmentation_pipeline('train'),
        transforms_val=get_augmentation_pipeline('val'),
        seed=42
    )

    dm.setup()

    train_dataset = dm._train_dataset
    val_dataset = dm._val_dataset

    # ================= REDUCE DATA =================
    train_subset = Subset(train_dataset, range(TRAIN_SAMPLES))
    val_subset = Subset(val_dataset, range(VAL_SAMPLES))

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Train samples: {len(train_subset)}")
    print(f"Val samples: {len(val_subset)}")
    print(f"Batches per epoch: {len(train_loader)}")

    # ================= MODEL =================
    model = AttentionUNet(4, 4).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = CombinedLoss(num_classes=4)

    # ================= TRAIN =================
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        model.train()
        total_loss = 0

        for images, masks in tqdm(train_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss, _ = loss_fn(outputs, masks)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Train Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(images)
                loss, _ = loss_fn(outputs, masks)

                val_loss += loss.item()

        print(f"Val Loss: {val_loss / len(val_loader):.4f}")

    print("\nDEBUG TRAINING COMPLETE ✅")

    # ================= SAVE MODEL =================
    project_root = Path(__file__).parent.parent
    save_dir = project_root / "models" / "saved"
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / "debug_model.pth"
    torch.save(model.state_dict(), model_path)

    print(f"\nModel saved at: {model_path}")


# 🔴 CRITICAL FIX
if __name__ == "__main__":
    main()