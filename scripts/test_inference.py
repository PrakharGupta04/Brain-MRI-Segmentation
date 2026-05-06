import sys
import os
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_registry import create_model
from utils.dataset_loader import BrainMRIDataset

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================= LOAD MODEL =================
model_name = os.environ.get("BRAIN_MRI_MODEL_NAME", "mobilenet_attention_unet")
model = create_model(model_name, 4, 4, pretrained=False).to(device)

checkpoint = torch.load("outputs/best_model.pth", map_location=device)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()
print(f"Model loaded successfully ({model_name})\n")

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
        print("Max edema class prob:", probs[:, 2, :, :].max().item())

        pred = torch.argmax(probs, dim=1)

    print(f"\nSample {i}")
    print("Ground Truth:", torch.unique(mask))
    print("Predicted   :", torch.unique(pred))