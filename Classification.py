import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import time

from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess, decode_predictions as mobilenet_decode
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess, decode_predictions as vgg_decode
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess, decode_predictions as resnet_decode
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess, decode_predictions as efficientnet_decode

st.set_page_config(page_title="Image Classification", layout="wide")
st.title("🧠 SmartVision AI - All Models Classification")

# Device Info
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    st.write(f"Running on GPU: {physical_devices}")
else:
    st.write("Running on CPU")

# Cache Models
@st.cache_resource
def load_model_by_name(name):
    if name == "MobileNetV2":
        return MobileNetV2(weights="imagenet")
    elif name == "VGG16":
        return VGG16(weights="imagenet")
    elif name == "ResNet50":
        return ResNet50(weights="imagenet")
    elif name == "EfficientNetB0":
        return EfficientNetB0(weights="imagenet")
    else:
        raise ValueError(f"Unknown model: {name}")

# Model Configs
model_configs = {
    "MobileNetV2": (mobilenet_preprocess, mobilenet_decode, "14 MB"),
    "VGG16": (vgg_preprocess, vgg_decode, "528 MB"),
    "ResNet50": (resnet_preprocess, resnet_decode, "98 MB"),
    "EfficientNetB0": (efficientnet_preprocess, efficientnet_decode, "29 MB")
}

# File Upload
uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, width=300)

    img_array = np.expand_dims(np.array(image), axis=0)  # batch dimension

    results_table = []
    time_table = []

    with col2:
        st.subheader("🔍 Model Predictions")

        # Run ALL models by default
        for model_name in model_configs.keys():
            preprocess_fn, decode_fn, size = model_configs[model_name]

            st.write(f"Loading **{model_name}** (~{size})...")

            model = load_model_by_name(model_name)
            input_tensor = preprocess_fn(img_array.copy())

            start = time.time()
            preds = model.predict(input_tensor)
            inference_time = time.time() - start

            decoded = decode_fn(preds, top=5)[0]

            top_label, top_score = decoded[0][1], decoded[0][2]

            results_table.append({
                "Model": model_name,
                "Top Prediction": top_label,
                "Confidence (%)": round(top_score * 100, 2)
            })

            time_table.append({
                "Model": model_name,
                "Inference Time (ms)": round(inference_time * 1000, 2)
            })

            st.markdown(f"**{model_name} Top-5 Predictions:**")
            for pred in decoded:
                st.write(f"{pred[1]} — {pred[2]*100:.2f}%")
            st.write("---")

    # Show summary tables
    st.markdown("### 📊 Summary Table (Top-1 Prediction)")
    st.dataframe(pd.DataFrame(results_table), use_container_width=True)

    st.markdown("### ⚡ Inference Time Comparison")
    st.dataframe(pd.DataFrame(time_table), use_container_width=True)
