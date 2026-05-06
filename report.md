📄 Dataset & Preprocessing Description (Report-Ready)

The dataset used in this project is the BraTS 2023 GLI (Brain Tumor Segmentation) dataset, which consists of multi-modal MRI scans of brain tumor patients. Each patient folder contains five 3D medical imaging files in NIfTI (.nii.gz) format: four MRI modalities—T1-weighted (T1n), T1-contrast enhanced (T1c), T2-weighted (T2w), and T2-FLAIR (T2f)—along with a corresponding segmentation mask (seg). These images are volumetric in nature, typically having dimensions around (240 × 240 × 155), where each slice represents a cross-section of the brain. The segmentation masks contain pixel-wise labels indicating different tumor regions such as edema, enhancing tumor, and non-enhancing tumor. However, this raw dataset is computationally heavy, unnormalized, and not directly suitable for training deep learning models, especially on low-resource systems.

To make the dataset suitable for training, a comprehensive preprocessing pipeline was implemented. First, the script dynamically identifies and loads all required modalities for each patient, handling flexible naming conventions in the dataset. Each MRI modality is then normalized using z-score normalization, considering only non-zero (brain tissue) regions to avoid background bias. The four modalities are stacked together to form a 4-channel volumetric input. Next, the 3D volumes are converted into 2D slices along the axial plane, significantly reducing computational complexity. To focus the learning process on relevant regions, only those slices that contain tumor pixels are retained (i.e., slices where the segmentation mask is non-zero). Each slice is then resized to a fixed resolution of 128 × 128 pixels to ensure uniformity across the dataset and compatibility with the model input size. Finally, the processed image slices and corresponding masks are saved as NumPy (.npy) files, and a metadata CSV file is generated to keep track of image-mask pairs along with patient and slice information.

After preprocessing, the dataset is transformed from a small number of large 3D volumes into a significantly larger set of lightweight 2D training samples. Specifically, from 1251 patients, a total of 81,437 valid slices were extracted, all correctly paired with their corresponding masks. The final dataset is structured into separate directories for images and masks, along with a metadata file ensuring consistency and traceability. This transformation greatly reduces memory requirements, speeds up training, and makes the dataset suitable for both local experimentation and cloud-based training environments like Kaggle. Additionally, by filtering out non-informative slices and normalizing the data, the preprocessing step improves the overall quality and learning efficiency of the deep learning model.

================================================================================
PREPROCESSING STATISTICS
================================================================================

Dataset Summary:
  Total Patients Found: 1251
  Successfully Processed: 1251
  Failed: 0
  Total Slices Extracted: 81437
  Average Slices/Patient: 65.1

Output Directories:
  Images: E:\Brain_MRI_DL\processed_data\images
  Masks: E:\Brain_MRI_DL\processed_data\masks
  Metadata CSV: E:\Brain_MRI_DL\processed_data\metadata.csv

Verification:
  Image files: 81437
  Mask files: 81437
  Metadata entries: 81437
  ✓ All files present and consistent!

  Dataset size: 57078
Image shape: torch.Size([4, 128, 128])
Mask shape: torch.Size([128, 128])
================================================================================

“The model was trained for only a few epochs on a small subset for debugging purposes. Due to class imbalance in medical imaging, the model initially learns to predict the dominant background class. Full training on GPU with more epochs resolves this issue.”
Input: torch.Size([1, 4, 128, 128])
Prediction: torch.Size([1, 128, 128])
Unique predicted labels: tensor([0])

Problems faced after doing the training 
After completing the initial 15 epochs of training, the model appeared to perform well based on aggregate metrics (around ~0.84 Dice score), but during inference we discovered a critical issue: the model was heavily biased toward predicting background and was consistently missing tumors, especially smaller or low-contrast regions. Debugging revealed that the predicted tumor probabilities were extremely low (often in the range of 0.001–0.03), indicating that although the model had learned general features, it lacked sensitivity to tumor classes due to severe class imbalance. Initially, we experimented with inference-level fixes such as threshold tuning, which temporarily improved detection but did not address the root cause. Further analysis showed that the loss function (a combination of Dice and CrossEntropy) was not correctly handling class imbalance—Dice was averaging improperly and included the background class, while class weights were not being effectively applied. We iteratively fixed the loss by (1) excluding background from Dice computation, (2) implementing proper weighted averaging, (3) reducing smoothing to increase sensitivity, and (4) correctly passing class weights to both Dice and CrossEntropy components. Instead of retraining from scratch, we then performed a targeted fine-tuning step by resuming from the previously trained model and training for 5 additional epochs with class-weighted loss, followed by a short 2-epoch refinement. This significantly improved tumor sensitivity, increasing confidence scores for previously missed regions. Finally, a mild thresholding step was applied during inference to capture borderline cases. The resulting model demonstrated consistent and reliable tumor detection across samples, including small tumors, without degrading overall performance—indicating a successful resolution of the class imbalance and sensitivity issues.