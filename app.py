import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# --------------------------
# Load TFLite model
# --------------------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="oral_cancer_model_optimized.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# --------------------------
# Prediction function
# --------------------------
def predict(image: Image.Image):
    # Resize to model input
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    confidence = float(output_data[0][0])
    prediction = 'Abnormal' if confidence > 0.5 else 'Normal'
    return prediction, confidence

# --------------------------
# Page layout + CSS
# --------------------------
st.set_page_config(page_title="Oral Cancer AI Detection", layout="centered", page_icon="🦷")

st.markdown("""
<style>
/* Full-screen light blue background */
html, body, .block-container, .main, .reportview-container {
    background-color: #dff0fb !important;
    height: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
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
    margin-bottom: 36px;
}

/* Main white card for uploader + results only */
.card {
    background: white;
    padding: 36px;
    border-radius: 26px;
    box-shadow: 0px 12px 30px rgba(0,0,0,0.08);
    margin-bottom: 24px;
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

/* Confidence score */
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

# --------------------------
# Page content
# --------------------------
st.markdown('<div class="title">🦷 Oral Cancer AI Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image of the oral cavity to receive an AI-based screening result. This tool is for educational purposes only and does not replace professional diagnosis.</div>', unsafe_allow_html=True)

# --------------------------
# Image uploader inside white card
# --------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload an oral cavity image (JPG or PNG)", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Analyze Image"):
        with st.spinner("Analyzing image..."):
            prediction, confidence = predict(image)
        if prediction == "Normal":
            st.markdown(f'<div class="normal"><b>Result:</b> {prediction}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="abnormal"><b>Result:</b> {prediction}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="score">Confidence Score: {confidence*100:.2f}%</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------
# Disclaimer
# --------------------------
st.markdown("""
<div class="disclaimer">
<b>Disclaimer:</b>
<ul style="margin-top:4px;">
<li>This application is a student research project and is not a medical device.</li>
<li>Results should not be used for diagnosis or treatment decisions.</li>
</ul>
</div>
""", unsafe_allow_html=True)
