import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Oral Cancer AI Detection",
    page_icon="🦷",
    layout="centered"
)

# -------------------------------
# CUSTOM CSS (DESIGN)
# -------------------------------
st.markdown(
    """
    <style>
        body {
            background-color: #eaf4fb;
        }

        .main {
            background-color: #eaf4fb;
        }

        h1, h2, h3, p, label {
            color: #3275a8;
            font-family: Arial, sans-serif;
        }

        .upload-box {
            background-color: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
            margin-top: 20px;
        }

        .disclaimer {
            font-size: 12px;
            color: #555555;
            margin-top: 25px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# TITLE SECTION
# -------------------------------
st.markdown("<h1>🦷 Oral Cancer AI Detection</h1>", unsafe_allow_html=True)
st.markdown(
    "<p>Upload a clear image of the mouth and let the AI analyze it.</p>",
    unsafe_allow_html=True
)

# -------------------------------
# LOAD TFLITE MODEL
# -------------------------------
@st.cache_resource
def load_model():
    model_path = "oral_cancer_model_optimized.tflite"

    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please upload oral_cancer_model_optimized.tflite")
        st.stop()

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------------
# IMAGE PREPROCESSING
# -------------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -------------------------------
# UPLOAD UI
# -------------------------------
st.markdown('<div class="upload-box">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📤 Upload a mouth image (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed_image = preprocess_image(image)

    interpreter.set_tensor(input_details[0]["index"], processed_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    prediction =
