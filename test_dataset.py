from utils.dataset_loader import BrainMRIDataset

dataset = BrainMRIDataset(
    data_dir="E:/Brain_MRI_DL/processed_data",
    metadata_csv="E:/Brain_MRI_DL/processed_data/metadata.csv",
    split="train"
)

print("Dataset size:", len(dataset))

img, mask = dataset[0]

print("Image shape:", img.shape)
print("Mask shape:", mask.shape)