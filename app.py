import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
from torchvision import transforms, models
import os
import gdown

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ======================
# STABIL
# ======================
torch.manual_seed(42)
np.random.seed(42)

st.set_page_config(page_title="Selada Classifier", layout="wide")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Bacterial', 'Fungal', 'Healthy']

# ======================
# TRANSFORM 
# ======================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ======================
# MODEL LINKS
# ======================
MODEL_LINKS = {
    ("DenseNet121", "Full Freeze"): "1pFPV_4_jbJRltbNLopm88nvIVKCjPWlb",
    ("DenseNet121", "Partial Unfreeze"): "1lYYa9AyCjC2KUBYxLVP2i6NYERwCKQOK",
    ("EfficientNetB0", "Full Freeze"): "1GlUrvPp_z7rXfR4qR6k1zBmu8ru7vHk3",
    ("EfficientNetB0", "Partial Unfreeze"): "1iAYtxo-cckCvQ7ROSh20pypVmEgs-tjs",
    ("MobileNetV3", "Full Freeze"): "1fofWMPyt81sOGDbubP10Ka1nxf2CAcCq",
    ("MobileNetV3", "Partial Unfreeze"): "1R9L3lY2GtCSxksKptwT7B1sbz2DGGRcz",
}

# ======================
# DOWNLOAD
# ======================
def download_model(file_id, output):
    os.makedirs("models", exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    return os.path.exists(output)

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model(arch, method):
    file_id = MODEL_LINKS[(arch, method)]
    path = f"models/{file_id}.pth"

    if not download_model(file_id, path):
        return None, None

    try:
        raw_state_dict = torch.load(path, map_location=device)
        
        state_dict = {}
        for k, v in raw_state_dict.items():
            state_dict[k.replace("module.", "").replace("model.", "")] = v

        actual_arch = arch
        if "features.conv0.weight" in state_dict:
            actual_arch = "DenseNet121"
        elif "features.0.0.weight" in state_dict:
            shape = state_dict["features.0.0.weight"].shape
            if shape[0] == 32:
                actual_arch = "EfficientNetB0"
            elif shape[0] == 16:
                actual_arch = "MobileNetV3"

        if actual_arch == "DenseNet121":
            model = models.densenet121(weights=None)
            if "classifier.0.weight" in state_dict and "classifier.3.weight" in state_dict:
                w1 = state_dict["classifier.0.weight"]
                w2 = state_dict["classifier.3.weight"]
                model.classifier = nn.Sequential(
                    nn.Linear(w1.shape[1], w1.shape[0]),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(w2.shape[1], w2.shape[0])
                )
            elif "classifier.weight" in state_dict:
                w = state_dict["classifier.weight"]
                model.classifier = nn.Linear(w.shape[1], w.shape[0])

        elif actual_arch == "EfficientNetB0":
            model = models.efficientnet_b0(weights=None)
            if "classifier.1.weight" in state_dict:
                w = state_dict["classifier.1.weight"]
                model.classifier[1] = nn.Linear(w.shape[1], w.shape[0])
            elif "classifier.weight" in state_dict:
                w = state_dict["classifier.weight"]
                model.classifier = nn.Linear(w.shape[1], w.shape[0])

        elif actual_arch == "MobileNetV3":
            model = models.mobilenet_v3_large(weights=None)
            if "classifier.0.weight" in state_dict:
                w0 = state_dict["classifier.0.weight"]
                model.classifier[0] = nn.Linear(w0.shape[1], w0.shape[0])
            if "classifier.3.weight" in state_dict:
                w3 = state_dict["classifier.3.weight"]
                model.classifier[3] = nn.Linear(w3.shape[1], w3.shape[0])
            elif "classifier.weight" in state_dict:
                w = state_dict["classifier.weight"]
                model.classifier = nn.Linear(w.shape[1], w.shape[0])

        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        return model, actual_arch

    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

# ======================
# SIDEBAR: FILTER GAMBAR
# ======================
st.sidebar.markdown("### ⚙️ Pengaturan Filter")
confidence_threshold = st.sidebar.slider(
    "Ambang Batas Keyakinan (Threshold)", 
    min_value=0.50, 
    max_value=0.99, 
    value=0.80, 
    step=0.01,
    help="Tingkatkan nilai ini untuk memblokir gambar asing/bukan selada. Jika model memprediksi gambar dengan keyakinan di bawah nilai ini, gambar akan ditolak."
)

# ======================
# UI UTAMA
# ======================
st.title("🌿 Klasifikasi Penyakit Daun Selada")

col1, col2 = st.columns(2)

with col1:
    selected_model = st.selectbox(
        "Arsitektur Target",
        ["DenseNet121", "EfficientNetB0", "MobileNetV3"]
    )

with col2:
    selected_method = st.selectbox(
        "Metode",
        ["Full Freeze", "Partial Unfreeze"]
    )

model, actual_arch = load_model(selected_model, selected_method)

if actual_arch and actual_arch != selected_model:
    st.warning(f"Terdeteksi link model tertukar! Arsitektur yang sedang digunakan (dibaca dari file) adalah: **{actual_arch}**")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

# ======================
# PREDIKSI & FILTERING
# ======================
if uploaded_file and model is not None:

    image = Image.open(uploaded_file).convert("RGB")

    colA, colB = st.columns(2)

    with colA:
        st.image(image, caption="Input Gambar", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(input_tensor)
        probs = torch.softmax(out, dim=1)[0]

    probs_np = probs.cpu().numpy()
    pred_class = int(np.argmax(probs_np))
    max_confidence = np.max(probs_np)

    # LOGIKA FILTER: Cek apakah confidence mencapai batas minimum
    if max_confidence < confidence_threshold:
        with colB:
            st.error("⚠️ Peringatan: Model tidak cukup yakin dengan gambar ini.")
            st.write(f"Keyakinan tertinggi hanya **{max_confidence:.4f}** (di bawah ambang batas **{confidence_threshold:.2f}**).")
            st.info("Kemungkinan besar ini adalah gambar acak, daun jenis lain (seperti tembakau), atau gambar terlalu buram/tidak fokus. Silakan unggah gambar daun selada yang jelas.")
        
        # Hentikan proses render grafik dan Grad-CAM jika gambar ditolak
        st.stop()
        
    else:
        with colB:
            st.success(f"Prediksi: {class_names[pred_class]}")
            st.write(f"Confidence: {max_confidence:.4f}")

        # ======================
        # GRAD-CAM
        # ======================
        st.divider()
        st.markdown("### 🔥 Visualisasi Model (Grad-CAM)")

        target_layers = [model.features[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(pred_class)]
        )[0]

        img_np = np.array(image.resize((224, 224))).astype(np.float32) / 255
        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        col1, col2 = st.columns(2, gap="small")

        with col1:
            st.image(image.resize((224, 224)), caption="Original Resized", use_container_width=True)

        with col2:
            st.image(cam_image, caption="Area Fokus (Grad-CAM)", use_container_width=True)

        # ======================
        # GRAFIK MODERN
        # ======================
        st.divider()
        st.markdown("### 📊 Probabilitas Klasifikasi")

        df = pd.DataFrame({
            "Kelas": class_names,
            "Probabilitas": probs_np,
            "Highlight": [
                "Prediksi" if i == pred_class else "Lainnya"
                for i in range(len(class_names))
            ]
        })

        chart = alt.Chart(df).mark_bar(
            cornerRadiusTopLeft=12,
            cornerRadiusTopRight=12
        ).encode(
            x=alt.X("Kelas", sort=None),
            y=alt.Y("Probabilitas", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "Highlight",
                scale=alt.Scale(
                    domain=["Prediksi", "Lainnya"],
                    range=["#00FF9C", "#555555"]
                ),
                legend=None
            ),
            tooltip=[
                alt.Tooltip("Kelas"),
                alt.Tooltip("Probabilitas", format=".4f")
            ]
        ).properties(
            height=320
        )

        st.altair_chart(chart, use_container_width=True)
