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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
# DOWNLOAD
# ======================
def download_model(file_id, output):
    os.makedirs("models", exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    return os.path.exists(output)

# ======================
# LOAD MODEL FINAL FIX
# ======================
@st.cache_resource
def load_model(arch, method):

    path = f"models/{arch}_{method}.pth"

    if not download_model(MODEL_LINKS[(arch, method)], path):
        return None

    try:
        raw_state_dict = torch.load(path, map_location=device)
        
        # Membersihkan prefix dari state_dict jika model di-save menggunakan DataParallel / wrapper lain
        state_dict = {}
        for k, v in raw_state_dict.items():
            clean_key = k.replace("module.", "").replace("model.", "")
            state_dict[clean_key] = v

        num_classes = len(class_names)

        # ======================
        # DENSENET
        # ======================
        if arch == "DenseNet121":
            model = models.densenet121(weights=None)
            in_features = model.classifier.in_features # Umumnya 1024
            
            # Deteksi Custom Classifier
            if "classifier.0.weight" in state_dict and "classifier.3.weight" in state_dict:
                hidden_out = state_dict["classifier.0.weight"].shape[0]
                model.classifier = nn.Sequential(
                    nn.Linear(in_features, hidden_out),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(hidden_out, num_classes)
                )
            else:
                model.classifier = nn.Linear(in_features, num_classes)

        # ======================
        # EFFICIENTNET
        # ======================
        elif arch == "EfficientNetB0":
            model = models.efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features # Umumnya 1280
            
            # Adaptasi berdasarkan kemungkinan modifikasi classifier saat training
            if "classifier.1.weight" in state_dict:
                model.classifier[1] = nn.Linear(in_features, num_classes)
            elif "classifier.weight" in state_dict:
                model.classifier = nn.Linear(in_features, num_classes)
            else:
                model.classifier[1] = nn.Linear(in_features, num_classes)

        # ======================
        # MOBILENET
        # ======================
        elif arch == "MobileNetV3":
            model = models.mobilenet_v3_large(weights=None)
            in_features = model.classifier[3].in_features # Umumnya 1280
            
            if "classifier.3.weight" in state_dict:
                model.classifier[3] = nn.Linear(in_features, num_classes)
            elif "classifier.weight" in state_dict:
                # Jika user mengganti keseluruhan nn.Sequential classifier dengan nn.Linear
                model.classifier = nn.Linear(960, num_classes)
            else:
                model.classifier[3] = nn.Linear(in_features, num_classes)

        # ======================
        # LOAD
        # ======================
        # Menggunakan strict=False agar aman dari key mismatch minor
        model.load_state_dict(state_dict, strict=False)

        model.to(device)
        model.eval()

        return model

    except Exception as e:
        st.error(f"Gagal load state_dict untuk {arch}: {e}")
        return None

# ======================
# UI
# ======================
st.title("🌿 Klasifikasi Penyakit Daun Selada")

col1, col2 = st.columns(2)

with col1:
    selected_model = st.selectbox(
        "Arsitektur",
        ["DenseNet121", "EfficientNetB0", "MobileNetV3"]
    )

with col2:
    selected_method = st.selectbox(
        "Metode",
        ["Full Freeze", "Partial Unfreeze"]
    )

model = load_model(selected_model, selected_method)

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

# ======================
# PREDIKSI
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
        st.image(image, caption="Original", use_container_width=True)

    with col2:
        st.image(cam_image, caption="Grad-CAM", use_container_width=True)

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
