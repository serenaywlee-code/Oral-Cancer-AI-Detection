import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Oral Cancer AI Detection",
    page_icon="🦷",
    layout="centered"
)

# ----------------------------
# CUSTOM CSS (DESIGN)
# ----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', 'Google Sans', 'Arial', sans-serif;
        background-color: white;
    }

    h1, h2, h3 {
        color: #3275a8;
    }

    p, label, span {
        color: #3275a8;
        font-size: 16px;
    }

    .stButton > button {
        background-color: #3275a8;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: 500;
    }

    .stFileUploader {
        border: 1px dashed #3275a8;
        border-radius: 8px;
        padding: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# HEADER
# ----------------------------
st.markdown("## 🦷 Oral Cancer AI Detection")
st.markdown(
    "Upload an image of the oral cavity to receive an **AI-based screening result**. "
    "This tool is for **educational purposes only** and does not replace professional diagnosis."
)

st.divider()

# ----------------------------
# LOAD TFLITE MODEL
# ----------------------------
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="oral_cancer_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------------------
# IMAGE PREPROCESSING + PREDICTION
# ----------------------------
def predict_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]["index"], image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])

    return float(prediction[0][0])

# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload an oral cavity image (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing image..."):
            confidence = predict_image(image)

        st.divider()

        if confidence >= 0.5:
            st.error(
                f"⚠️ **Potential abnormality detected**\n\n"
                f"Confidence: **{confidence*100:.1f}%**"
            )
        else:
            st.success(
                f"✅ **Appears normal**\n\n"
                f"Confidence: **{(1-confidence)*100:.1f}%**"
            )

# ----------------------------
# FOOTER / DISCLAIMER
# ----------------------------
st.divider()
st.markdown(
    """
    **Disclaimer:**  
    This application is a **student research project** and is **not a medical device**.  
    Results should **not** be used for diagnosis or treatment decisions.
    """
)
