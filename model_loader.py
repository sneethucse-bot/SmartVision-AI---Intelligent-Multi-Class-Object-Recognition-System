import streamlit as st
import tensorflow as tf
from ultralytics import YOLO

@st.cache_resource
def load_classification_model(path):
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)