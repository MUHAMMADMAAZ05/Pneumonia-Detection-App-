import os
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from torchvision import transforms, models

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Pneumonia Detection",
    page_icon="🫁",
    layout="wide"
)

# =========================
# UI STYLE
# =========================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eff6ff, #f8fafc, #ecfeff);
}

.title-box {
    background: linear-gradient(90deg, #2563eb, #0f766e);
    padding: 28px;
    border-radius: 18px;
    text-align: center;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
}

.card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
}

.result-normal {
    background: #dcfce7;
    color: #166534;
    padding: 24px;
    border-radius: 14px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
}

.result-pneumonia {
    background: #fee2e2;
    color: #991b1b;
    padding: 24px;
    border-radius: 14px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
}

.info-box {
    background: #e0f2fe;
    padding: 14px;
    border-radius: 12px;
    color: #075985;
    border: 1px solid #7dd3fc;
}

.warning-box {
    background: #fff7ed;
    padding: 14px;
    border-radius: 12px;
    color: #9a3412;
    border: 1px solid #fdba74;
}
</style>
""", unsafe_allow_html=True)

# =========================
# MODEL ARCHITECTURE
# =========================

class ResNet18Embedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()

        self.embedding = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding(features)
        return embeddings

# =========================
# PATHS
# =========================

MODEL_PATH = "fewshot_resnet18_focal_pneumonia.pth"
TRAIN_DIR = os.path.join("chest_xray", "train")

classes = ["NORMAL", "PNEUMONIA"]

# =========================
# IMAGE TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet18Embedding().to(device)

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    model.eval()

    return model, device

model, device = load_model()

# =========================
# SUPPORT IMAGE LOADER
# =========================

def get_class_images(class_folder, max_images=5):
    image_files = []

    if not os.path.exists(class_folder):
        return image_files

    for file in os.listdir(class_folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_files.append(os.path.join(class_folder, file))

    return image_files[:max_images]


@st.cache_resource
def get_prototypes():
    embeddings = []

    for cls in classes:
        cls_folder = os.path.join(TRAIN_DIR, cls)
        image_paths = get_class_images(cls_folder, max_images=5)

        if len(image_paths) == 0:
            st.error(
                f"No support images found in {cls_folder}. "
                "Keep chest_xray/train/NORMAL and chest_xray/train/PNEUMONIA folders."
            )
            st.stop()

        class_embeddings = []

        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = model(img)

            class_embeddings.append(emb.squeeze(0))

        class_embeddings = torch.stack(class_embeddings)
        class_prototype = class_embeddings.mean(dim=0)
        embeddings.append(class_prototype)

    prototypes = torch.stack(embeddings)

    return prototypes

# =========================
# PREDICTION FUNCTION
# =========================

def predict_image(image):
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    prototypes = get_prototypes()

    with torch.no_grad():
        query_embedding = model(image_tensor)

        distances = torch.cdist(query_embedding, prototypes)
        scores = -distances
        probabilities = torch.softmax(scores, dim=1)

        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item()

    return classes[predicted_idx], confidence, probabilities[0].cpu().numpy()

# =========================
# HEADER
# =========================

st.markdown("""
<div class="title-box">
    <h1>🫁 AI Pneumonia Detection System</h1>
    <p>Hybrid ResNet-18 + Transfer Learning + Prototypical Few-Shot Learning + Focal Loss</p>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.header("📌 Project Details")
    st.write("**Model:** ResNet-18")
    st.write("**Learning:** Transfer Learning")
    st.write("**Few-Shot:** Prototypical Network")
    st.write("**Loss:** Focal Loss")
    st.write("**Classes:** NORMAL / PNEUMONIA")

    st.info("Upload a chest X-ray image to classify it as NORMAL or PNEUMONIA.")

    st.warning("Academic project only. Not for real medical diagnosis.")

# =========================
# MAIN UI
# =========================

left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📤 Upload Chest X-ray")

    uploaded_file = st.file_uploader(
        "Upload X-ray image",
        type=["jpg", "jpeg", "png"]
    )

    st.markdown("""
    <div class="info-box">
        File name does not matter. Upload any chest X-ray image from your PC.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 Model Information")
    st.write("This system predicts whether a chest X-ray is:")
    st.write("✅ NORMAL")
    st.write("🫁 PNEUMONIA")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# IMAGE + PREDICTION
# =========================

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.markdown("---")

    img_col, pred_col = st.columns([1, 1])

    with img_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🖼 Uploaded X-ray")
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with pred_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔍 Prediction")

        if st.button("Predict Result", use_container_width=True):
            result, confidence, probs = predict_image(image)

            if result == "PNEUMONIA":
                st.markdown(
                    f"""
                    <div class="result-pneumonia">
                        🫁 PNEUMONIA<br>
                        Confidence: {confidence * 100:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.warning("The model detected pneumonia-like patterns.")
            else:
                st.markdown(
                    f"""
                    <div class="result-normal">
                        ✅ NORMAL<br>
                        Confidence: {confidence * 100:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.success("The model classified this X-ray as normal.")

            st.markdown("### Probability Scores")
            st.progress(float(probs[0]), text=f"NORMAL: {probs[0] * 100:.2f}%")
            st.progress(float(probs[1]), text=f"PNEUMONIA: {probs[1] * 100:.2f}%")

        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# FOOTER
# =========================

st.markdown("---")
st.caption("⚠️ This application is for academic and research purposes only.")