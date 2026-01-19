import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG (MUST BE FIRST)
# --------------------------------------------------
st.set_page_config(
    page_title="Oral Cancer AI Detection",
    page_icon="🦷",
    layout="centered"
)

# --------------------------------------------------
# CUSTOM CSS (DESIGN)
# --------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        background-color: white;
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, p, label {
        color: #3275a8 !important;
        font-family: 'Inter', sans-serif;
    }

    .welcome-box {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        background-color: #f9fbfd;
    }

    .result-box {
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
        background-color: #eef6fb;
        font-weight: 500;
    }

    .footer {
        font-size: 12px;
        color: #6b7280;
        text-align: center;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL (CACHED)
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("oral_cancer_model.keras")

model = load_model()

# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# --------------------------------------------------
# UI CONTENT
# --------------------------------------------------
st.markdown("""
<div class="welcome-box">
    <h1>🦷 Oral Cancer AI Detection</h1>
    <p>
    Upload an oral cavity image and this AI model will analyze it for signs
    associated with oral cancer.  
    <br><br>
    This tool is for <b>educational and research purposes only</b> and is not a medical diagnosis.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an oral cavity image (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)[0][0]

    if prediction > 0.5:
        result = "⚠️ Higher likelihood of oral cancer features detected"
    else:
        result = "✅ Lower likelihood of oral cancer features detected"

    st.markdown(f"""
    <div class="result-box">
        <p>{result}</p>
        <p><b>Model confidence:</b> {prediction:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<div class="footer">
    Built with Streamlit · AI-assisted health research project
</div>
""", unsafe_allow_html=True)
