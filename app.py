import streamlit as st

st.set_page_config(
    page_title="SmartVision AI",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 SmartVision AI")
st.sidebar.success("Select a page from the sidebar")

st.markdown("""
Welcome to **SmartVision AI** — an Intelligent Multi-Class Object Recognition System.

Use the sidebar to navigate between:
- 🧠 Image Classification
- 🎯 Object Detection
- 📊 Model Performance
- 📹 Live Webcam
""")