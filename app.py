import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Oral Cancer AI Detection",
    page_icon="🦷",
    layout="centered"
)

# ---------------- STYLING ----------------
st.markdown("""
<style>
/* Entire page background */
html, body, .main {
    background-color: #e6f2fb;
}

/* Title */
.title {
    font-size: 40px;
    font-weight: 700;
    color: #3275a8;
    text-align: center;
    margin-bottom: 8px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #4a4a4a;
    margin-bottom: 30px;
}

/* White card */
.card {
    background: white;
    padding: 32px;
    border-radius: 22px;
    box-shadow: 0px 8px 22px rgba(0,0,0,0.08);
}

/* Normal result */
.result-normal {
    background: #f0fdf4;
    border-left: 6px solid #22c55e;
    padding: 18px;
    border-radius: 12px;
    color: #14532d;
}

/* Abnormal result */
.result-abnormal {
    background: #fff7ed;
    border-left: 6px solid #f59e0b;
    padding: 18px;
    border-radius: 12px;
    color: #7c2d12;
}

/* Disclaimer */
.disclaimer {
    background: rgba(254, 202, 202, 0.45);
    border-left: 6px solid #dc2626;
    padding: 18px;
    border-radius: 14px;
    margin-top: 28px;
    color: #b91c1c;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>🦷 Oral Cancer AI Detection</div>", unsafe_allow_html=True)
st.markdown("""
<div class='subtitle'>
Upload an image of the oral cavity to receive an AI-based screening result.<br>
This tool is for educational purposes only and does not replace professional diagnosis.
</div>
""", unsafe_allow_html=True)

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(
        model_path="oral_cancer_model_optimized.tflite"
    )
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

# ---------------- UPLOAD CARD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### Upload an oral cavity image (JPG or PNG)", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        processed = preprocess_image(image)
        interpreter.set_tensor(input_details[0]["index"], processed)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        prediction = float(output[0][0])

    if prediction > 0.5:
        st.markdown("""
        <div class='result-abnormal'>
        <h4>⚠️ Abnormality Detected</h4>
        The AI has detected potential abnormalities. Please consult a healthcare professional immediately.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='result-normal'>
        <h4>✅ Normal Result</h4>
        No abnormalities detected. Please consult a professional for confirmation.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"**Confidence score:** `{prediction:.2f}`")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- DISCLAIMER ----------------
st.markdown("""
<div class='disclaimer'>
<strong>Disclaimer</strong><br>
• This application is a student research project and is not a medical device.<br>
• Results must not be used for diagnosis or treatment decisions.
</div>
""", unsafe_allow_html=True)
