import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
import os
import pandas as pd
import altair as alt
import gdown

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ======================
# FIX RANDOM (STABIL)
# ======================
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="Selada Classifier", layout="wide")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Bacterial', 'Fungal', 'Healthy']

# ======================
# TRANSFORM
# ======================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ======================
# MODEL LINKS
# ======================
MODEL_LINKS = {
    ("DenseNet121", "Full Freeze"): "1pFPV_4_jbJRltbNLopm88nvIVKCjPWlb",
    ("DenseNet121", "Partial Unfreeze"): "1iAYtxo-cckCvQ7ROSh20pypVmEgs-tjs",

    ("EfficientNetB0", "Full Freeze"): "1fofWMPyt81sOGDbubP10Ka1nxf2CAcCq",
    ("EfficientNetB0", "Partial Unfreeze"): "1GlUrvPp_z7rXfR4qR6k1zBmu8ru7vHk3",

    ("MobileNetV3", "Full Freeze"): "1R9L3lY2GtCSxksKptwT7B1sbz2DGGRcz",
    ("MobileNetV3", "Partial Unfreeze"): "1lYYa9AyCjC2KUBYxLVP2i6NYERwCKQOK",
}

# ======================
# DOWNLOAD MODEL
# ======================
def download_model(file_id, output):
    os.makedirs("models", exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    if not os.path.exists(output):
        return False

    # validasi ukuran file
    size = os.path.getsize(output)
    if size < 5_000_000:
        return False

    return True

# ======================
# BUILD MODEL (FALLBACK SAJA)
# ======================
def build_model(arch):

    if arch == "DenseNet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 3)

    elif arch == "EfficientNetB0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)

    elif arch == "MobileNetV3":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 3)

    return model

# ======================
# LOAD MODEL (FINAL)
# ======================
@st.cache_resource
def load_model(arch, method):

    path = f"models/{arch}_{method}.pth"

    if not download_model(MODEL_LINKS[(arch, method)], path):
        return None

    # ======================
    # 🔥 1. PRIORITAS FULL MODEL
    # ======================
    try:
        model = torch.load(path, map_location=device)

        if hasattr(model, "eval"):
            model.to(device)
            model.eval()
            return model
    except:
        pass

    # ======================
    # 🔥 2. STATE_DICT (HARUS MATCH)
    # ======================
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

        else:
            return None

        model = build_model(arch)

        # ❗ WAJIB MATCH (tidak pakai strict=False)
        model.load_state_dict(state_dict, strict=True)

        model.to(device)
        model.eval()

        return model

    except:
        return None

# ======================
# UI
# ======================
st.title("🌿 Klasifikasi Penyakit Daun Selada")

col1, col2 = st.columns(2)

with col1:
    selected_model = st.selectbox(
        "Arsitektur",
        ["DenseNet121","EfficientNetB0","MobileNetV3"]
    )

with col2:
    selected_method = st.selectbox(
        "Metode",
        ["Full Freeze","Partial Unfreeze"]
    )

model = load_model(selected_model, selected_method)

# ======================
# UPLOAD
# ======================
uploaded_file = st.file_uploader("Upload gambar", type=["jpg","png","jpeg"])

# ======================
# INFERENSI
# ======================
if uploaded_file and model is not None:

    image = Image.open(uploaded_file).convert("RGB")

    colA, colB = st.columns(2)

    with colA:
        st.image(image, caption="Input", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]

    probs_np = probs.cpu().numpy()
    pred_class = np.argmax(probs_np)

    with colB:
        st.success(class_names[pred_class])
        st.write(f"Confidence: {np.max(probs_np):.4f}")

    # ======================
    # GRAD-CAM
    # ======================
    st.divider()
    st.subheader("Grad-CAM")

    try:
        target_layers = [model.features[-1]]
    except:
        target_layers = [list(model.children())[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(pred_class)]
    )[0]

    cam_image = show_cam_on_image(
        np.array(image.resize((224,224))).astype(np.float32)/255,
        grayscale_cam,
        use_rgb=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, use_container_width=True)

    with col2:
        st.image(cam_image, use_container_width=True)

elif uploaded_file and model is None:
    st.warning("Model tidak dapat digunakan")
