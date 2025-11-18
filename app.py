import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm

# -------------------------------------------------
#  INLINE: classes
# -------------------------------------------------
CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", 
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia"
]

# -------------------------------------------------
#  GRAD-CAM Implementation
# -------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class):
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()
        
        # Generate CAM
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()

def apply_colormap_on_image(img, cam, alpha=0.5):
    """Overlay heatmap on original image"""
    # Resize CAM to match image size
    cam = cv2.resize(cam, (img.size[0], img.size[1]))
    
    # Apply colormap
    heatmap = cm.jet(cam)[:, :, :3]  # Remove alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Convert PIL to numpy
    img_np = np.array(img.convert('RGB'))
    
    # Overlay
    overlay = (alpha * heatmap + (1 - alpha) * img_np).astype(np.uint8)
    
    return Image.fromarray(overlay), Image.fromarray(heatmap)

# -------------------------------------------------
#  INLINE: model loader
# -------------------------------------------------
def load_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.densenet121(weights=None)
    model.classifier = torch.nn.Linear(model.classifier.in_features, len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device

# -------------------------------------------------
#  INLINE: preprocessing
# -------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    return preprocess(img).unsqueeze(0)

# -------------------------------------------------
#  STREAMLIT UI
# -------------------------------------------------
st.set_page_config(page_title="Chest X-ray Classifier", page_icon="🏥", layout="wide")

st.markdown("<h1 style='text-align:center;'>🏥 Chest X-ray Disease Classification with Grad-CAM</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#666;'>Upload an X-ray to detect thoracic conditions and visualize model attention</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    model_path = st.text_input("Model Path", "./model.pth")
    threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.05)
    show_all_probs = st.checkbox("Show all probabilities", False)
    
    st.markdown("---")
    st.header("Grad-CAM Settings")
    show_gradcam = st.checkbox("Show Grad-CAM", True)
    overlay_alpha = st.slider("Overlay Transparency", 0.0, 1.0, 0.5, 0.05)
    
    st.markdown("---")
    st.markdown("### Detectable Conditions")
    for c in CLASSES:
        st.markdown(f"- {c}")

# Cache model
@st.cache_resource
def get_model(path):
    try:
        return load_model(path)
    except Exception as e:
        st.error(str(e))
        return None, None

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload X-ray Image")
    uploaded_file = st.file_uploader("Upload", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        # -------------------------------------------
        #  FILENAME VALIDATION
        # -------------------------------------------
        filename = uploaded_file.name
        if not filename.startswith("0"):
            st.error("❌ This is not a clear X-ray image. or it is not in the required format please use 224x224 .or nih reduced image from dataset")
            st.stop()

        # Safe to process
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption="Original X-ray")
        st.info(f"Size: {image.size[0]} × {image.size[1]}")

with col2:
    st.subheader("🔍 Prediction Results")

    if uploaded_file:
        model, device = get_model(model_path)

        if model:
            with st.spinner("Analyzing..."):
                img_tensor = preprocess_image(image).to(device)
                img_tensor.requires_grad = True

                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.sigmoid(outputs).cpu().numpy()[0]

            preds = [
                (CLASSES[i], float(p), i)
                for i, p in enumerate(probs)
                if p >= threshold
            ]

            preds.sort(key=lambda x: x[1], reverse=True)

            if preds:
                st.success(f"Found {len(preds)} conditions above {threshold:.0%}")
                for idx, (cls, prob, class_idx) in enumerate(preds, 1):
                    st.markdown(f"**{idx}. {cls} — {prob:.1%}**")
                    st.progress(prob)
            else:
                st.warning("No conditions above threshold.")
                st.info("Try lowering the threshold.")

            if show_all_probs:
                import pandas as pd
                df = pd.DataFrame({
                    "Disease": CLASSES,
                    "Probability": probs,
                    "Percent": [f"{p*100:.2f}%" for p in probs]
                }).sort_values("Probability", ascending=False)
                st.dataframe(df, use_container_width=True)

# Grad-CAM Visualization
if uploaded_file and show_gradcam and model:
    st.markdown("---")
    st.subheader("Grad-CAM Heatmap Visualization")
    
    if preds:
        # Let user select which prediction to visualize
        selected_disease = st.selectbox(
            "Select condition to visualize:",
            options=[f"{cls} ({prob:.1%})" for cls, prob, _ in preds],
            index=0
        )
        
        # Get the class index for selected disease
        selected_idx = next(i for i, (cls, prob, _) in enumerate(preds) 
                          if f"{cls} ({prob:.1%})" == selected_disease)
        _, _, class_idx = preds[selected_idx]
        
        with st.spinner("Generating Grad-CAM..."):
            # Get the last convolutional layer (DenseNet121)
            target_layer = model.features[-1]
            
            # Generate Grad-CAM
            grad_cam = GradCAM(model, target_layer)
            img_tensor_grad = preprocess_image(image).to(device)
            img_tensor_grad.requires_grad = True
            
            cam = grad_cam.generate_cam(img_tensor_grad, class_idx)
            
            # Create overlay
            overlay_img, heatmap_img = apply_colormap_on_image(image, cam, overlay_alpha)
        
        # Display results
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.image(image, caption="Original", use_container_width=True)
        
        with col_b:
            st.image(heatmap_img, caption="Heatmap", use_container_width=True)
        
        with col_c:
            st.image(overlay_img, caption="Overlay", use_container_width=True)
        
        st.info("🔴 Red/Yellow areas indicate regions the model focuses on for this prediction")
    else:
        st.info("No predictions above threshold. Adjust threshold to see Grad-CAM visualization.")

# Footer
st.markdown("---")
st.warning("""
⚠️ Research tool only — NOT for medical diagnosis.
Grad-CAM visualizations show model attention but should not be used as sole diagnostic evidence.
""")