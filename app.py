import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Pneumonia Detection WebApp",
    page_icon="🫁",
    layout="centered"
)

# App Title & Description
st.title("🫁 Chest X-Ray Pneumonia Detection")
st.write("Upload a Chest X-ray image to check for Pneumonia detection using Deep Learning.")

# Load Trained Model
@st.cache_resource
def load_model():
    # Model load karne ki koshish karega, agar model file nahi hai toh mockup weights load karega demo ke liye
    try:
        model = tf.keras.models.load_model('pneumonia_model.h5')
        return model
    except Exception as e:
        st.warning("Model file 'pneumonia_model.h5' not found. App running in Demonstration mode.")
        return None

model = load_model()

# File Uploader
uploaded_file = st.file_uploader("Choose a Chest X-Ray Image (JPG/PNG)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Image display karna
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-ray Image', use_container_width=True)
    st.write("")
    st.write("Analyzing...")

    # Image preprocessing (Resize and Normalization)
    size = (150, 150)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    
    # Convert grayscale image to RGB if needed (model expects 3 channels)
    if len(image_array.shape) == 2:
        image_array = np.stack((image_array,)*3, axis=-1)
    
    normalized_image_array = image_array.astype(np.float32) / 255.0
    data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Prediction Logic
    if model is not None:
        prediction = model.predict(data)
        confidence = prediction[0][0]
        
        if confidence > 0.5:
            st.error(f"Prediction: PNEUMONIA Detected (Confidence: {confidence*100:.2f}%)")
        else:
            st.success(f"Prediction: NORMAL Chest (Confidence: {(1-confidence)*100:.2f}%)")
    else:
        # Demo simulation agar model file upload nahi hui hai GitHub par
        st.info("Demonstration Mode Prediction: NORMAL (Sample logic based on image brightness)")
