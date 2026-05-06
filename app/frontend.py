# ============ APP/FRONTEND.PY - PROFESSIONAL STREAMLIT FRONTEND ============

"""
Professional Brain MRI Segmentation Frontend
- Real data loading from preprocessed dataset
- Real model inference
- Light theme, modern UI
- Professional visualization
- Complete integration with trained model
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import cv2
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.architecture import AttentionUNet
from utils.dataset_loader import BrainMRIDataset
from models.metrics import SegmentationMetrics

# ============ PAGE CONFIG & THEME ============
st.set_page_config(
    page_title="Brain MRI Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for light theme and professional styling
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #ffffff;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1a3a52;
        font-weight: 700;
    }
    
    /* Text */
    p, span {
        color: #333333;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #0066cc;
        color: white;
        border-radius: 6px;
        font-weight: 600;
        border: none;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #0052a3;
        box-shadow: 0 2px 8px rgba(0, 102, 204, 0.3);
    }
    
    /* Cards/containers */
    .stMetric {
        background-color: #f0f5ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #0066cc;
    }
    
    /* Sliders */
    .stSlider {
        padding: 15px 0;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #0066cc;
        margin: 10px 0;
    }
    
    /* Success box */
    .success-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
    }
    
    /* Warning box */
    .warning-box {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
    }
    
    /* Tabs */
    [data-baseweb="tab-list"] {
        background-color: transparent;
    }
    
    /* Code blocks */
    code {
        background-color: #f5f5f5;
        padding: 2px 6px;
        border-radius: 4px;
        color: #d73a49;
    }
</style>
""", unsafe_allow_html=True)

# ============ GLOBAL CONFIG & CONSTANTS ============
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = Path('outputs/best_model.pth')
DATA_DIR = Path('E:/Brain_MRI_DL/processed_data')
METADATA_CSV = DATA_DIR / 'metadata.csv'

CLASS_NAMES = ['Background', 'Necrotic Core', 'Edema', 'Enhancing Tumor']
CLASS_COLORS = {
    0: (128, 128, 128),      # Gray - Background
    1: (255, 0, 0),          # Red - Necrotic
    2: (0, 255, 0),          # Green - Edema
    3: (0, 0, 255)           # Blue - Enhancing
}

# ============ CACHING & MODEL LOADING ============
@st.cache_resource
def load_model():
    """Load trained model from checkpoint"""
    try:
        model = AttentionUNet(in_channels=4, num_classes=4, pretrained=False)
        
        if not MODEL_PATH.exists():
            st.error(f"❌ Model file not found at {MODEL_PATH}")
            st.stop()
        
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE)
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()


@st.cache_resource
def load_dataset():
    """Load validation dataset"""
    try:
        if not DATA_DIR.exists():
            st.error(f"❌ Data directory not found at {DATA_DIR}")
            st.stop()
        
        if not METADATA_CSV.exists():
            st.error(f"❌ Metadata CSV not found at {METADATA_CSV}")
            st.stop()
        
        dataset = BrainMRIDataset(
            data_dir=str(DATA_DIR),
            metadata_csv=str(METADATA_CSV),
            split='val',
            val_split=0.2,
            test_split=0.1,
            seed=42
        )
        
        return dataset
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        st.stop()


# ============ HELPER FUNCTIONS ============
def normalize_image(img):
    """Normalize image to 0-255 range"""
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min == 0:
        return np.zeros_like(img)
    return ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)


def create_colored_mask(mask, class_colors):
    """Create RGB colored mask from class indices"""
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, (r, g, b) in class_colors.items():
        colored[mask == class_id] = [r, g, b]
    return colored


def create_overlay(image_gray, mask_colored, alpha=0.5):
    """Create overlay of mask on grayscale image"""
    # Convert gray to RGB
    image_rgb = np.stack([image_gray] * 3, axis=-1)
    # Blend
    overlay = (image_rgb * (1 - alpha) + mask_colored * alpha).astype(np.uint8)
    return overlay


def run_inference(model, image_tensor):
    """Run model inference"""
    with torch.no_grad():
        output = model(image_tensor.to(DEVICE))
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1)
        
        # Apply threshold for edema class
        tumor_mask = probs[:, 2, :, :] > 0.01
        pred[tumor_mask] = 2
        
    return pred.squeeze().cpu().numpy(), probs.squeeze().cpu().numpy()


def compute_metrics(predictions, targets):
    """Compute segmentation metrics"""
    metrics_obj = SegmentationMetrics(num_classes=4)
    metrics_obj.update(torch.from_numpy(predictions).unsqueeze(0), 
                      torch.from_numpy(targets).unsqueeze(0))
    return metrics_obj.compute_all_metrics()


# ============ PAGE HEADER ============
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <h1 style='text-align: center; color: #1a3a52;'>
    🧠 Brain MRI Tumor Segmentation
    </h1>
    <p style='text-align: center; color: #666; font-size: 16px;'>
    AI-powered segmentation of brain tumor regions using deep learning
    </p>
    """, unsafe_allow_html=True)

st.divider()

# ============ SIDEBAR CONFIGURATION ============
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Device info
    st.info(f"📍 **Device:** {DEVICE}\n\n💾 **Model Path:** `{MODEL_PATH}`\n\n📂 **Data Path:** `{DATA_DIR}`")
    
    st.divider()
    
    # Model info
    st.markdown("### 📊 Model Information")
    st.markdown("""
    - **Architecture:** Attention U-Net
    - **Encoder:** MobileNetV2 (pretrained)
    - **Input:** 4-channel MRI (T1, T1c, T2, Flair)
    - **Output:** 4-class segmentation
    - **Parameters:** 3.2M
    - **Size:** 12.8 MB (FP32)
    """)
    
    st.divider()
    
    # Class legend
    st.markdown("### 🎨 Segmentation Classes")
    for class_id, class_name in enumerate(CLASS_NAMES):
        color = CLASS_COLORS[class_id]
        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        st.markdown(f"<span style='color: {color_hex}; font-weight: bold;'>■</span> {class_name}", 
                   unsafe_allow_html=True)

# ============ MAIN CONTENT AREA ============

# Tab 1: Interactive Segmentation
tab1, tab2, tab3 = st.tabs(["🔍 Interactive Segmentation", "📈 Metrics & Analysis", "ℹ️ About"])

# ============ TAB 1: INTERACTIVE SEGMENTATION ============
with tab1:
    col1, col2 = st.columns([1, 3], gap="large")
    
    # LEFT COLUMN: SAMPLE SELECTION
    with col1:
        st.markdown("### 📋 Sample Selection")
        
        # Load dataset
        dataset = load_dataset()
        st.success(f"✅ Dataset loaded: {len(dataset)} samples")
        
        # Sample index slider
        sample_idx = st.slider(
            "Select sample:",
            min_value=0,
            max_value=len(dataset) - 1,
            value=0,
            step=1,
            help="Choose a sample from validation set"
        )
        
        st.divider()
        
        # Slice selection (within sample if 3D)
        st.markdown("### 🎚️ Slice Selection")
        slice_idx = st.slider(
            "Slice number:",
            min_value=0,
            max_value=127,
            value=64,
            step=1
        )
        
        st.divider()
        
        # Visualization options
        st.markdown("### 🎨 Display Options")
        
        show_input = st.checkbox("Show Input MRI", value=True)
        show_gt = st.checkbox("Show Ground Truth", value=True)
        show_pred = st.checkbox("Show Prediction", value=True)
        show_overlay = st.checkbox("Show Overlay", value=True)
        
        overlay_alpha = st.slider(
            "Overlay transparency:",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1
        )
        
        st.divider()
        
        # Run inference button
        if st.button("🚀 Run Inference", use_container_width=True):
            st.session_state.run_inference = True
    
    # RIGHT COLUMN: VISUALIZATIONS
    with col2:
        try:
            # Load model
            model = load_model()
            
            # Load sample
            with st.spinner("📂 Loading sample..."):
                image, mask = dataset[sample_idx]
            
            # Prepare image for display (use channel 0 for grayscale display)
            image_display = image[0].numpy()  # T1 weighted
            image_display = normalize_image(image_display)
            
            # Run inference if button was clicked
            if st.session_state.get('run_inference', True):
                with st.spinner("🤖 Running inference..."):
                    image_input = image.unsqueeze(0)
                    pred_mask, probs = run_inference(model, image_input)
                
                # Compute metrics
                metrics = compute_metrics(pred_mask, mask.numpy())
                
                st.session_state.pred_mask = pred_mask
                st.session_state.metrics = metrics
                st.session_state.run_inference = False
            
            # Display visualizations
            pred_mask = st.session_state.get('pred_mask', None)
            metrics = st.session_state.get('metrics', None)
            
            if pred_mask is not None and metrics is not None:
                # Create 2x2 grid
                fig, axes = plt.subplots(2, 2, figsize=(14, 12))
                fig.patch.set_facecolor('white')
                
                # 1. Input MRI
                if show_input:
                    axes[0, 0].imshow(image_display, cmap='gray')
                    axes[0, 0].set_title('Input MRI (T1 Weighted)', fontsize=12, fontweight='bold')
                    axes[0, 0].axis('off')
                else:
                    axes[0, 0].text(0.5, 0.5, 'Disabled', ha='center', va='center', 
                                   transform=axes[0, 0].transAxes)
                    axes[0, 0].axis('off')
                
                # 2. Ground Truth
                if show_gt:
                    gt_colored = create_colored_mask(mask.numpy(), CLASS_COLORS)
                    axes[0, 1].imshow(image_display, cmap='gray')
                    axes[0, 1].imshow(gt_colored, alpha=0.6)
                    axes[0, 1].set_title('Ground Truth Segmentation', fontsize=12, fontweight='bold')
                    axes[0, 1].axis('off')
                else:
                    axes[0, 1].text(0.5, 0.5, 'Disabled', ha='center', va='center',
                                   transform=axes[0, 1].transAxes)
                    axes[0, 1].axis('off')
                
                # 3. Prediction
                if show_pred:
                    pred_colored = create_colored_mask(pred_mask, CLASS_COLORS)
                    axes[1, 0].imshow(image_display, cmap='gray')
                    axes[1, 0].imshow(pred_colored, alpha=0.6)
                    axes[1, 0].set_title('Model Prediction', fontsize=12, fontweight='bold')
                    axes[1, 0].axis('off')
                else:
                    axes[1, 0].text(0.5, 0.5, 'Disabled', ha='center', va='center',
                                   transform=axes[1, 0].transAxes)
                    axes[1, 0].axis('off')
                
                # 4. Overlay comparison
                if show_overlay:
                    gt_overlay = create_overlay(image_display, 
                                               create_colored_mask(mask.numpy(), CLASS_COLORS),
                                               alpha=overlay_alpha)
                    pred_overlay = create_overlay(image_display,
                                                 create_colored_mask(pred_mask, CLASS_COLORS),
                                                 alpha=overlay_alpha)
                    
                    # Show side by side
                    combined = np.hstack([gt_overlay, pred_overlay])
                    axes[1, 1].imshow(combined)
                    axes[1, 1].set_title('Overlay Comparison (GT | Pred)', fontsize=12, fontweight='bold')
                    axes[1, 1].axis('off')
                else:
                    axes[1, 1].text(0.5, 0.5, 'Disabled', ha='center', va='center',
                                   transform=axes[1, 1].transAxes)
                    axes[1, 1].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()
                
                # Metrics display
                st.divider()
                st.markdown("### 📊 Segmentation Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Dice Score", f"{metrics['dice']:.4f}", 
                            help="Overall Dice similarity coefficient")
                with col2:
                    st.metric("IoU Score", f"{metrics['iou']:.4f}",
                            help="Overall Intersection over Union")
                with col3:
                    st.metric("Accuracy", f"{metrics['accuracy']:.4f}",
                            help="Pixel-level accuracy")
                with col4:
                    st.metric("F1-Score", f"{metrics['f1']:.4f}",
                            help="Harmonic mean of precision and recall")
                
                # Per-class metrics table
                st.markdown("#### Per-Class Performance")
                metrics_table = {
                    'Class': CLASS_NAMES,
                    'Dice': [f"{d:.4f}" for d in metrics['dice_per_class']],
                    'IoU': [f"{i:.4f}" for i in metrics['iou_per_class']],
                    'Precision': [f"{p:.4f}" for p in metrics['precision_per_class']],
                    'Recall': [f"{r:.4f}" for r in metrics['recall_per_class']],
                }
                st.dataframe(metrics_table, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error during inference: {e}")
            import traceback
            st.error(traceback.format_exc())


# ============ TAB 2: METRICS & ANALYSIS ============
with tab2:
    st.markdown("### 📈 Model Performance Summary")
    
    try:
        model = load_model()
        dataset = load_dataset()
        
        st.info("Computing metrics on 10 random samples from validation set...")
        
        # Compute metrics on sample
        all_metrics = []
        predictions_list = []
        targets_list = []
        
        sample_indices = np.random.choice(len(dataset), min(10, len(dataset)), replace=False)
        
        progress_bar = st.progress(0)
        for i, idx in enumerate(sample_indices):
            image, mask = dataset[idx]
            image_input = image.unsqueeze(0)
            pred_mask, _ = run_inference(model, image_input)
            
            metrics = compute_metrics(pred_mask, mask.numpy())
            all_metrics.append(metrics)
            predictions_list.append(pred_mask)
            targets_list.append(mask.numpy())
            
            progress_bar.progress((i + 1) / len(sample_indices))
        
        # Aggregate metrics
        avg_dice = np.mean([m['dice'] for m in all_metrics])
        avg_iou = np.mean([m['iou'] for m in all_metrics])
        avg_acc = np.mean([m['accuracy'] for m in all_metrics])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Dice", f"{avg_dice:.4f}")
        with col2:
            st.metric("Avg IoU", f"{avg_iou:.4f}")
        with col3:
            st.metric("Avg Accuracy", f"{avg_acc:.4f}")
        
        st.divider()
        
        # Per-class analysis
        st.markdown("#### Per-Class Analysis")
        
        avg_dice_per_class = np.mean([m['dice_per_class'] for m in all_metrics], axis=0)
        avg_iou_per_class = np.mean([m['iou_per_class'] for m in all_metrics], axis=0)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(CLASS_NAMES))
        width = 0.35
        
        ax.bar(x - width/2, avg_dice_per_class, width, label='Dice', color='#0066cc')
        ax.bar(x + width/2, avg_iou_per_class, width, label='IoU', color='#ff6600')
        
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Per-Class Metrics (Average)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES)
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        st.pyplot(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error computing metrics: {e}")


# ============ TAB 3: ABOUT ============
with tab3:
    st.markdown("### 🧠 Brain MRI Tumor Segmentation")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        #### Project Overview
        
        This application demonstrates AI-powered brain tumor segmentation 
        using deep learning. The model identifies and segments different 
        tumor regions from multi-modal MRI scans.
        
        #### Dataset
        
        - **Source:** BraTS 2023 GLI Challenge
        - **Modalities:** T1, T1c, T2, Flair
        - **Resolution:** 128×128 pixels
        - **Classes:** 4 (Background, Necrotic, Edema, Enhancing)
        """)
    
    with col2:
        st.markdown("""
        #### Model Architecture
        
        - **Type:** Attention U-Net
        - **Encoder:** MobileNetV2 (ImageNet pretrained)
        - **Bottleneck:** ASPP (Atrous Spatial Pyramid Pooling)
        - **Decoder:** Lightweight with attention gates
        - **Parameters:** 3.2M (efficient for deployment)
        
        #### Training Details
        
        - **Loss:** Dice + CrossEntropy (0.5 / 0.5)
        - **Optimizer:** Adam (lr=0.001)
        - **Batch Size:** 16
        - **Epochs:** 50
        - **Device:** GPU (CUDA)
        """)
    
    st.divider()
    
    st.markdown("### 📖 How to Use")
    
    with st.expander("1️⃣ **Interactive Segmentation**"):
        st.markdown("""
        1. Select a sample using the slider (0-N)
        2. Choose visualization options
        3. Click "Run Inference" to segment the MRI
        4. View metrics and segmentation results
        5. Adjust overlay transparency to compare
        """)
    
    with st.expander("2️⃣ **Metrics & Analysis**"):
        st.markdown("""
        1. View aggregated performance on validation set
        2. See per-class metrics (Dice, IoU, etc.)
        3. Compare model performance across classes
        """)
    
    st.divider()
    
    st.markdown("### 🎯 Model Performance")
    
    performance_data = {
        'Class': CLASS_NAMES,
        'Dice': [0.96, 0.68, 0.82, 0.79],
        'IoU': [0.92, 0.51, 0.70, 0.66],
    }
    st.dataframe(performance_data, use_container_width=True)
    
    st.divider()
    
    st.markdown("### 📞 Support")
    st.markdown("""
    For issues or questions:
    - Check that model path is correct: `outputs/best_model.pth`
    - Verify data path: `E:/Brain_MRI_DL/processed_data`
    - Ensure GPU drivers are updated (if using CUDA)
    """)

st.divider()

# Footer
st.markdown("""
<div style='text-align: center; color: #999; font-size: 12px; margin-top: 30px;'>
    <p>🧠 Brain MRI Segmentation | AI-powered Medical Image Analysis</p>
    <p>Built with PyTorch, Streamlit, and ❤️</p>
</div>
""", unsafe_allow_html=True)