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

    # basic sanity check
    size = os.path.getsize(output)
    if size < 5_000_000:  # <5MB biasanya bukan weight valid
        return False

    return True

# ======================
# UTIL: ambil state_dict apapun formatnya
# ======================
def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint
    return None

# ======================
# UTIL: infer arsitektur dari key
# ======================
def infer_arch_from_keys(state_dict):
    k = next(iter(state_dict.keys()))
    # DenseNet
    if "denseblock" in k:
        return "densenet"
    # MobileNetV3 biasanya punya fitur conv awal "features.0.0.weight"
    if k.startswith("features.0.0.") or ".block." in k:
        return "mobilenet_v3"
    # EfficientNetB0 biasanya "features.0.weight" / "features.1.0.block"
    if k.startswith("features.0.") or ".blocks." in k:
        return "efficientnet_b0"
    return None

# ======================
# UTIL: sesuaikan classifier dari weight checkpoint
# ======================
def adapt_classifier_from_state_dict(model, arch, state_dict, num_classes):
    """
    Tujuan: bikin layer classifier punya shape yang sama dengan checkpoint,
    supaya strict=True bisa lolos.
    """
    # cari key weight classifier terakhir
    # DenseNet: "classifier.weight"
    # EfficientNet: "classifier.1.weight"
    # MobileNetV3: "classifier.3.weight"
    weight_key = None
    for k in state_dict.keys():
        if k.endswith("classifier.weight") or k.endswith("classifier.1.weight") or k.endswith("classifier.3.weight"):
            weight_key = k

    # fallback: cari key yang ukurannya (num_classes, *)
    if weight_key is None:
        for k, v in state_dict.items():
            if hasattr(v, "shape") and len(v.shape) == 2 and v.shape[0] == num_classes:
                weight_key = k
                break

    if weight_key is None:
        return model  # tidak ketemu, biarkan default

    w = state_dict[weight_key]
    out_features, in_features = w.shape

    if arch == "densenet":
        model.classifier = nn.Linear(in_features, out_features)

    elif arch == "efficientnet_b0":
        # classifier = Sequential(Dropout, Linear)
        if isinstance(model.classifier, nn.Sequential) and len(model.classifier) >= 2:
            model.classifier[1] = nn.Linear(in_features, out_features)
        else:
            model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, out_features))

    elif arch == "mobilenet_v3":
        # classifier = Sequential(..., Linear) biasanya index 3
        if isinstance(model.classifier, nn.Sequential) and len(model.classifier) >= 4:
            model.classifier[3] = nn.Linear(in_features, out_features)
        else:
            model.classifier = nn.Sequential(
                nn.Linear(in_features, 1280),
                nn.Hardswish(),
                nn.Dropout(0.2),
                nn.Linear(1280, out_features),
            )

    return model

# ======================
# BUILD BASE MODEL
# ======================
def build_base_model(arch):
    if arch == "densenet":
        m = models.densenet121(weights=None)
        return m
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        return m
    if arch == "mobilenet_v3":
        m = models.mobilenet_v3_large(weights=None)
        return m
    return None

# ======================
# LOAD MODEL (ROBUST)
# ======================
@st.cache_resource
def load_model(arch, method):

    path = f"models/{arch}_{method}.pth"

    if not download_model(MODEL_LINKS[(arch, method)], path):
        return None

    try:
        state_dict = torch.load(path, map_location=device)

        # ======================
        # 🔥 CARI LAYER CLASSIFIER OTOMATIS
        # ======================
        fc_weight_key = None

        for key in state_dict.keys():
            if "classifier" in key and "weight" in key:
                fc_weight_key = key
                break

        if fc_weight_key is None:
            # fallback cari fc
            for key in state_dict.keys():
                if "fc" in key and "weight" in key:
                    fc_weight_key = key
                    break

        if fc_weight_key is None:
            st.error("Tidak ditemukan layer classifier")
            return None

        weight = state_dict[fc_weight_key]

        out_features = weight.shape[0]
        in_features = weight.shape[1]

        # ======================
        # BUILD MODEL
        # ======================
        if arch == "DenseNet121":
            model = models.densenet121(weights=None)
            model.classifier = nn.Linear(in_features, out_features)

        elif arch == "EfficientNetB0":
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(in_features, out_features)

        elif arch == "MobileNetV3":
            model = models.mobilenet_v3_large(weights=None)
            model.classifier[3] = nn.Linear(in_features, out_features)

        # ======================
        # LOAD
        # ======================
        model.load_state_dict(state_dict, strict=True)

        model.to(device)
        model.eval()

        return model

    except Exception as e:
        st.error(f"Gagal load: {e}")
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
        out = model(input_tensor)
        probs = torch.softmax(out, dim=1)[0]

    probs_np = probs.cpu().numpy()
    pred_class = int(np.argmax(probs_np))

    with colB:
        st.success(class_names[pred_class])
        st.write(f"Confidence: {np.max(probs_np):.4f}")

    # Grad-CAM
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

    c1, c2 = st.columns(2)
    with c1:
        st.image(image, use_container_width=True)
    with c2:
        st.image(cam_image, use_container_width=True)

elif uploaded_file and model is None:
    st.warning("Model tidak dapat digunakan (arsitektur/weight tidak cocok)")
