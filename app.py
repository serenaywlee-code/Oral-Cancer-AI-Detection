import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Oral Cancer AI Detection",
    page_icon="🦷",
    layout="centered"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
/* Entire app background */
.stApp {
    background-color: #d0e7ff;
}

/* Remove black container */
.block-container {
    background-color: transparent !important;
    padding-top: 2rem;
}

/* Text styling */
html, body, [class*="css"]  {
    font-family: 'Arial', sans-serif;
    color: #3275a8;
}

/* Upload card */
.upload-card {
    background-color: white;
    padding: 2rem;
    border-radius: 16px;
    border: 2px dashed #3275a8;
    margin-top: 1.5rem;
}

/* Buttons */
.stButton > button {
    background-color: #3275a8;
    color: white;
    border-radius: 10px;
    padding: 0.5em 1.5em;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.markdown("<h1 style='text-align:center;'>🦷 Oral Cancer AI Detection</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Upload a mouth image and let the AI analyze it.</p>",
    unsafe_allow_html=True
)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    model_path = "oral_cancer_model.tflite"

    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Make sure `oral_cancer_model.tflite` is uploaded.")
        st.stop()

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------------------ UPLOAD AREA ------------------
st.markdown("<div class='upload-card'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a mouth image",
    type=["jpg", "jpeg", "png"]
)

st.markdown("</div>", unsafe_allow_html=True)

# ------------------ PREDICTION ------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    prediction = np.argmax(output)

    if prediction == 1:
        st.error("⚠️ Possible signs of oral cancer detected.")
    else:
        st.success("✅ No signs of oral cancer detected.")

# ------------------ DISCLAIMER ------------------
st.markdown("---")
st.markdown(
    "<small>This tool is for educational purposes only and is not a medical diagnosis.</small>",
    unsafe_allow_html=True
)
