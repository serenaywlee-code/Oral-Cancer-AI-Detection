import streamlit as st
import os
import gdown
import numpy as np
from PIL import Image
import tensorflow as tf

# ---- Page Setup ----
st.set_page_config(page_title="Oral Cancer AI Screening", layout="centered")
st.title("🦷 Oral Cancer AI Screening Tool")

st.markdown("""
Upload a clear photo of the inside of the mouth.  
This tool uses a trained AI model to flag **possible signs** associated with oral cancer.  
⚠️ **This is NOT a medical diagnosis.**
""")

# ---- Download Model from Google Drive if needed ----

MODEL_PATH = "oral_cancer_model.tflite"
DRIVE_LINK = "https://drive.google.com/uc?id=1dADkE996bl3iCs6_-ECIcLaIGG_7W1eZ"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model ...")
    gdown.download(DRIVE_LINK, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully!")

# ---- Load TFLite Model ----

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---- Prediction Function ----
def predict(image: Image.Image):
    # Resize & normalize
    image = image.resize((224, 224))
    array = np.array(image) / 255.0
    array = np.expand_dims(array.astype(np.float32), axis=0)

    interpreter.set_tensor(input_details[0]["index"], array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    # Binary classification
    score = float(output[0])
    label = "Possible Signs" if score > 0.5 else "Likely Normal"
    confidence = score if score > 0.5 else (1.0 - score)
    return label, confidence * 100

# ---- Upload & Predict ----
uploaded_file = st.file_uploader("Upload a mouth image:", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image ..."):
        label, confidence = predict(image)

    if label == "Possible Signs":
        st.error(f"⚠️ {label} ({confidence:.2f}% confidence)")
    else:
        st.success(f"✅ {label} ({confidence:.2f}% confidence)")

    st.caption("This is a research screening tool, not a medical diagnosis.")
