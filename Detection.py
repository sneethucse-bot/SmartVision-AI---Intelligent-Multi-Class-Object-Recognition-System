# Detection.py

import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Paths
MODEL_DIR = "models/detection"
MODEL_PATH = os.path.join(MODEL_DIR, "yolov8_best.pt")

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load YOLO model
@st.cache_resource  # caches the model in Streamlit
def load_yolo_model(path):
    if not os.path.exists(path):
        st.warning(f"{path} not found. Downloading pretrained YOLOv8 model...")
        # Download pretrained YOLOv8n model and save it as yolov8_best.pt
        model = YOLO("yolov8n.pt")  # YOLOv8 nano pretrained
        model.save(path)
        st.success("Pretrained YOLOv8 model downloaded and saved!")
    else:
        model = YOLO(path)
    return model

# Load the model
model = load_yolo_model(MODEL_PATH)

# Streamlit UI
st.title("🔍 Object Detection with YOLOv8")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=700)  

    # Run detection
    results = model.predict(image)

    # Render results
    annotated_image = results[0].plot()  # returns numpy array
    st.image(annotated_image, caption="Detection Results", width=700)  
