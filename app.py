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

# ---------------- STYLES ----------------
st.markdown("""
<style>
/* Page background */
body, .block-container, .main {
    background-color: #dff0fb !important;  /* Light blue background */
}

/* Title */
.title {
    font-size: 40px;
    font-weight: 700;
    color: #3275a8;
    text-align: center;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #4a4a4a;
    margin-bottom: 36px;
}

/* Results */
.normal {
    background: #f0fdf4;
    border-left: 6px solid #22c55e;
    padding: 16px;
    border-radius: 12px;
    color: #14532d;
    margin-top: 16px;
}

.abnormal {
    background: #fff7ed;
    border-left: 6px solid #f59e0b;
    padding: 16px;
    border-radius: 12px;
    color: #7c2d12;
    margin-top: 16px;
}

/* Risk score */
.score {
    margin-top: 14px;
    font-weight: 600;
    color: #374151;
}

/* Disclaimer */
.disclaimer {
    background: rgba(254, 202, 202, 0.45);
    border-left: 6px solid #dc2626;
    padding: 18px;
    border-radius: 14px;
    color: #b91c1c;
    font-size: 15px;
    margin-top: 32px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>🦷 Oral Cancer AI Detection</div>", unsafe_allow_html=True)
st.markdown("""
<div class='subtitle'>
st.markdown(
    "<h3 style='color:#3275a8;'>Upload an oral cavity image (JPG or PNG)</h3>",
    unsafe_allow_html=True
)
This tool is for educational purposes only and does not replace professional diagnosis.
</div>

""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
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

def preprocess(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0).astype(np.float32)

# ---------------- UPLOAD + RESULTS CARD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("### Upload an oral cavity image (JPG or PNG)")

uploaded = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded:
    image = Image.open(uploaded)
    st.image(image, use_container_width=True)

    with st.spinner("Analyzing image..."):
        x = preprocess(image)
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        raw_pred = float(interpreter.get_tensor(output_details[0]["index"])[0][0])

    risk_score = int(raw_pred * 100)

    if risk_score >= 71:
        st.markdown(f"""
        <div class='abnormal'>
        <strong>🔴 High Risk Detected</strong><br>
        The AI detected features associated with possible abnormalities.
        </div>
        """, unsafe_allow_html=True)
    elif risk_score >= 41:
        st.markdown(f"""
        <div class='abnormal'>
        <strong>🟡 Moderate Risk Detected</strong><br>
        Some irregular features were identified.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='normal'>
        <strong>🟢 Low Risk</strong><br>
        No significant abnormalities detected.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        f"<div class='score'>Risk Score: {risk_score} / 100</div>",
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- DISCLAIMER ----------------
st.markdown("""
<div class='disclaimer'>
<strong>Disclaimer</strong><br>
• This application is a student research project and is not a medical device.<br>
• Results must not be used for diagnosis or treatment decisions.
</div>
""", unsafe_allow_html=True)
