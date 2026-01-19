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
import streamlit as st

st.set_page_config(
    page_title="Oral Health AI",
    page_icon="🦷",
    layout="centered"
)

st.markdown("""
<style>
    .stApp {
        background-color: #fafafa;
    }
    h1, h2, h3 {
        color: #4a6fa5;
    }
    .info-box {
        background-color: #eef4ff;
        padding: 1rem;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)
st.markdown(
    "<h1 style='text-align:center;'>🦷 Oral Health AI Assistant</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; color:#666;'>A gentle AI screening tool for oral health awareness</p>",
    unsafe_allow_html=True
)
st.markdown("""
<div class="info-box">
<b>Hi there! 👋</b><br><br>
I’m an AI tool trained to look for <i>visual patterns</i> in mouth images that <i>may</i> be associated with oral cancer.<br><br>
⚠️ This is <b>not</b> a medical diagnosis — just an early awareness tool.
</div>
""", unsafe_allow_html=True)
st.markdown("### Upload an image")

uploaded_file = st.file_uploader(
    "Choose a clear photo of the mouth area",
    type=["jpg", "jpeg", "png"]
)
if uploaded_file:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    with col2:
        st.markdown("### 🧠 AI Analysis")
        st.write("Looking for visual patterns… please wait 💭")
if prediction == 1:
    st.warning(
        "⚠️ The AI noticed **visual patterns that may need attention**.\n\n"
        "This does not mean you have oral cancer.\n"
        "Please consider seeing a dental professional."
    )
else:
    st.success(
        "✅ No concerning visual patterns were detected.\n\n"
        "If you notice any changes or symptoms, a dentist can help."
    )
st.markdown("### What should I do next?")
st.write("""
- Keep monitoring your oral health
- Maintain regular dental check-ups
- Seek professional advice if you’re concerned
""")
st.markdown("---")
st.markdown(
    "<small>Built by Serena • For education & awareness only • Not a diagnosis 💙</small>",
    unsafe_allow_html=True
)
