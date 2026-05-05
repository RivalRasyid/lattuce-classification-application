import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
import os
import pandas as pd
import altair as alt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

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
# MODEL PATH
# ======================
MODEL_PATHS = {
    ("DenseNet121", "Full Freeze"): "models/best_densenet121_freeze.pth",
    ("DenseNet121", "Partial Unfreeze"): "models/best_densenet121_partial.pth",

    ("EfficientNetB0", "Full Freeze"): "models/best_efficientnetb0_freeze.pth",
    ("EfficientNetB0", "Partial Unfreeze"): "models/best_efficientnetb0_partial_unfreeze.pth",

    ("MobileNetV3", "Full Freeze"): "models/best_mobilenetv3_freeze.pth",
    ("MobileNetV3", "Partial Unfreeze"): "models/best_mobilenetv3_partial.pth",
}

# ======================
# BUILD MODEL
# ======================
def build_model(arch):
    if arch == "DenseNet121":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 1280),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1280, 3)
        )

    elif arch == "EfficientNetB0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 3)
        )

    elif arch == "MobileNetV3":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, 3)

    return model

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model(arch, method):
    model = build_model(arch)
    path = MODEL_PATHS[(arch, method)]
    full_path = os.path.join(os.path.dirname(__file__), path)

    model.load_state_dict(torch.load(full_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ======================
# UI HEADER
# ======================
st.markdown("## 🌿 Klasifikasi Penyakit Daun Selada")
st.caption("Sistem deteksi penyakit berbasis deep learning")

col1, col2 = st.columns(2)

with col1:
    selected_model = st.selectbox(
        "Pilih Arsitektur",
        ["DenseNet121","EfficientNetB0","MobileNetV3"]
    )

with col2:
    selected_method = st.selectbox(
        "Pilih Metode",
        ["Full Freeze","Partial Unfreeze"]
    )

# MODEL
model = load_model(selected_model, selected_method)
validator_model = load_model("EfficientNetB0", "Partial Unfreeze")

uploaded_file = st.file_uploader("📤 Upload gambar", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    colA, colB = st.columns([1,1.2])

    # ======================
    # INPUT
    # ======================
    with colA:
        st.markdown("### 📷 Gambar Input")
        st.image(image, use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    # ======================
    # PREDIKSI
    # ======================
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]

    probs_np = probs.cpu().numpy()
    sorted_probs = np.sort(probs_np)[::-1]

    max_prob = sorted_probs[0]
    second_prob = sorted_probs[1]
    pred_class = np.argmax(probs_np)

    # ======================
    # THRESHOLD
    # ======================
    if selected_method == "Full Freeze":
        CONF_THRESHOLD = 0.55
    else:
        CONF_THRESHOLD = 0.75

    # ======================
    # VALIDATOR
    # ======================
    with torch.no_grad():
        val_output = validator_model(input_tensor)
        val_probs = torch.softmax(val_output, dim=1)[0].cpu().numpy()

    val_class = np.argmax(val_probs)
    val_conf = np.max(val_probs)

    valid = (
        max_prob > CONF_THRESHOLD and
        (max_prob - second_prob) > 0.1 and
        val_conf > 0.6 and
        val_class == pred_class
    )

    if not valid:
        st.error("❌ Non-lettuce (bukan daun selada)")
        st.write(f"Confidence utama: {max_prob:.4f}")
        st.write(f"Validator: {val_conf:.4f}")
        st.stop()

    # ======================
    # HASIL
    # ======================
    with colB:
        st.markdown("### 🧠 Hasil Prediksi")
        st.success(f"{class_names[pred_class]}")
        st.write(f"Confidence: {max_prob:.4f}")

    # ======================
    # GRAD-CAM
    # ======================
    st.divider()
    st.markdown("### 🔥 Visualisasi Model (Grad-CAM)")
    st.caption("Area berwarna merah menunjukkan bagian daun yang paling diperhatikan model")

    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(pred_class)]
    )[0]

    img_np = np.array(image.resize((224,224))).astype(np.float32)/255
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col1:
        st.image(image, caption="Original", width=400)

    with col2:
        st.image(cam_image, caption="Grad-CAM", width=400)

    with col3:
        st.empty()
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
        y=alt.Y("Probabilitas", scale=alt.Scale(domain=[0,1])),
        color=alt.Color(
            "Highlight",
            scale=alt.Scale(
                domain=["Prediksi","Lainnya"],
                range=["#00FF9C","#555555"]
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