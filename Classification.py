import streamlit as st
from PIL import Image
from utils.model_loader import load_classification_model
from utils.classifier import preprocess_image, predict
from utils.benchmarking import measure_inference_time

st.title("🧠 Image Classification")

uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_column_width=True)

    img_array = preprocess_image(image)

    models = ["VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0"]
    num_classes = 26  # number of classes in your dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for name in models:
        model = load_classification_model(name, num_classes, device=device)
        results = predict(model, img_array)
        inference_time = measure_inference_time(model, img_array)

        st.subheader(name)
        st.write(f"Inference Time: {inference_time*1000:.2f} ms")
        for label, score in results:
            st.write(f"{label} — {score:.2%}")