import streamlit as st
import torch
import torch.nn as nn
import numpy as np
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
        state_dict = torch.load(path, map_location=device)

        # ======================
        # 🔥 AUTO DETECT CLASSIFIER
        # ======================
        classifier_keys = [k for k in state_dict.keys() if "classifier" in k and "weight" in k]

        if len(classifier_keys) == 0:
            st.error("Classifier tidak ditemukan")
            return None

        # urutkan berdasarkan index
        classifier_keys.sort()

        # ======================
        # BUILD BASE MODEL
        # ======================
        if arch == "DenseNet121":
            model = models.densenet121(weights=None)

        elif arch == "EfficientNetB0":
            model = models.efficientnet_b0(weights=None)

        elif arch == "MobileNetV3":
            model = models.mobilenet_v3_large(weights=None)

        # ======================
        # 🔥 BUILD CLASSIFIER DINAMIS
        # ======================
        layers = []

        for key in classifier_keys:
            w = state_dict[key]
            out_f, in_f = w.shape

            layers.append(nn.Linear(in_f, out_f))

        # kalau lebih dari 1 layer → jadikan Sequential
        if len(layers) > 1:
            classifier = nn.Sequential(*layers)
        else:
            classifier = layers[0]

        # assign ke model
        if arch == "DenseNet121":
            model.classifier = classifier

        elif arch == "EfficientNetB0":
            model.classifier = nn.Sequential(nn.Dropout(0.2), classifier)

        elif arch == "MobileNetV3":
            model.classifier = nn.Sequential(
                nn.Linear(layers[0].in_features, layers[0].in_features),
                nn.Hardswish(),
                nn.Dropout(0.2),
                classifier
            )

        # ======================
        # LOAD STATE_DICT
        # ======================
        model.load_state_dict(state_dict, strict=False)

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
    # GRADCAM
    # ======================
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
    st.warning("Model tidak dapat digunakan")
