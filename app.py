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
body {
    background-color: #eaf4fb;
}
.main {
    background-color: #eaf4fb;
}
.title {
    font-size: 40px;
    font-weight: 700;
    color: #3275a8;
    text-align: center;
}
.subtitle {
    text-align: center;
    font-size: 17px;
    color: #4a4a4a;
}
.card {
    background: white;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0px 8px 24px rgba(0,0,0,0.08);
    margin-top: 30px;
}
.result-normal {
    background: #ecfdf5;
    border-left: 6px solid #22c55e;
    padding: 20px;
    border-radius: 12px;
}
.result-abnormal {
    background: #fff7ed;
    border-left: 6px solid #f59e0b;
    padding: 20px;
    border-radius: 12px;
}
.disclaimer {
    background: #fef2f2;
    border-left: 6px solid #ef4444;
    padding: 20px;
    border-radius: 12px;
    margin-top: 30px;
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
    return tf.lite.Interpreter(model_path="oral_cancer_model_optimized.tflite")

interpreter = load_model()
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

# ---------------- UPLOAD CARD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### Upload an oral cavity image (JPG or PNG)")

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        processed = preprocess_image(image)
        interpreter.set_tensor(input_details[0]["index"], processed)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        prediction = float(output[0][0])

    if prediction > 0.5:
        st.markdown("""
        <div class='result-abnormal'>
        <h3>⚠️ Abnormality Detected</h3>
        The AI has detected potential abnormalities. Please consult a healthcare professional immediately.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='result-normal'>
        <h3>✅ Normal Result</h3>
        No abnormalities detected. Please consult a professional for confirmation.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"**Confidence score:** `{prediction:.2f}`")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- DISCLAIMER ----------------
st.markdown("""
<div class='disclaimer'>
<h4>Disclaimer</h4>
• This application is a student research project and is not a medical device.<br>
• Results must not be used for diagnosis or treatment decisions.
</div>
""", unsafe_allow_html=True)
