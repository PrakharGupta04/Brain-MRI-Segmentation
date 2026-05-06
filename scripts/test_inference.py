import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.architecture import AttentionUNet
from utils.dataset_loader import BrainMRIDataset

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================= LOAD MODEL =================
model = AttentionUNet(4, 4).to(device)

checkpoint = torch.load("outputs/best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()
print("Model loaded successfully\n")

# ================= LOAD DATA =================
dataset = BrainMRIDataset(
    data_dir="E:/Brain_MRI_DL/processed_data",
    metadata_csv="E:/Brain_MRI_DL/processed_data/metadata.csv",
    split="val"
)

print(f"Dataset size: {len(dataset)}\n")

# ================= TEST MULTIPLE SAMPLES =================
print("===== CHECKING MULTIPLE SAMPLES =====")

for i in range(10):
    img, mask = dataset[i]

    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)

        # Optional: soften predictions (helps debug)
        probs = torch.softmax(output, dim=1)
        print("Max tumor prob:", probs[:, 2, :, :].max().item())

        pred = torch.argmax(probs, dim=1)
        tumor_mask = probs[:, 2, :, :] > 0.01
        pred[tumor_mask] = 2

    print(f"\nSample {i}")
    print("Ground Truth:", torch.unique(mask))
    print("Predicted   :", torch.unique(pred))