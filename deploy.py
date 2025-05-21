import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO
import numpy as np
import tempfile
import os

# Title
st.title("🚧 Road Pothole Detection using YOLOv8")
st.markdown("Upload a road image, and the model will detect potholes.")

# Load model

def load_model():
    model = YOLO("yolov8s_pothole2/weights/best.pt")  # Replace with your trained model path
    return model

model = load_model()

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Run detection
if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_path = temp_file.name
        image.save(temp_path)

    # Run YOLO inference
    results = model(temp_path)

    # Display results
    st.subheader("🔍 Detection Results")
    res_img = results[0].plot()  # Annotated image as NumPy array
    st.image(res_img, caption="Pothole Detection", use_column_width=True)

    # Optional: Show detection labels and confidence
    st.write("### 📋 Detections")
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        st.write(f"Class: {model.names[cls]}, Confidence: {conf:.2f}")

    # Cleanup temp file
    os.remove(temp_path)
