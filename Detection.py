import streamlit as st
import numpy as np
from PIL import Image
from utils.model_loader import load_yolo_model
from utils.detector import run_detection
from config import DEFAULT_CONFIDENCE

st.title("🎯 Object Detection")

uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
confidence = st.slider("Confidence Threshold", 0.1, 0.9, DEFAULT_CONFIDENCE)

if uploaded:
    image = Image.open(uploaded)
    image_np = np.array(image)

    model = load_yolo_model("models/detection/yolov8_best.pt")
    result_img = run_detection(model, image_np, confidence)

    st.image(result_img, use_column_width=True)