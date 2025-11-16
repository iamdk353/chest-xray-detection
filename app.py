import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image

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

st.markdown("<h1 style='text-align:center;'>🏥 Chest X-ray Disease Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#666;'>Upload an X-ray to detect thoracic conditions</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    model_path = st.text_input("Model Path", "./model.pth")
    threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.05)
    show_all_probs = st.checkbox("Show all probabilities", False)

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
        st.image(image, use_container_width=True)
        st.info(f"Size: {image.size[0]} × {image.size[1]}")

with col2:
    st.subheader("🔍 Prediction Results")

    if uploaded_file:
        model, device = get_model(model_path)

        if model:
            with st.spinner("Analyzing..."):
                img_tensor = preprocess_image(image).to(device)

                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.sigmoid(outputs).cpu().numpy()[0]

            preds = [
                (CLASSES[i], float(p))
                for i, p in enumerate(probs)
                if p >= threshold
            ]

            preds.sort(key=lambda x: x[1], reverse=True)

            if preds:
                st.success(f"Found {len(preds)} conditions above {threshold:.0%}")
                for i, (cls, prob) in enumerate(preds, 1):
                    st.markdown(f"**{i}. {cls} — {prob:.1%}**")
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

# Footer
st.markdown("---")
st.warning("""
⚠️ Research tool only — NOT for medical diagnosis.
""")
