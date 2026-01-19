import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Oral Cancer Screening (Research Tool)",
    layout="centered"
)

st.title("🦷 Oral Cancer Image Screening Tool")
st.write(
    "Upload an image of the mouth. "
    "This tool uses a machine learning model to **flag possible signs** of oral cancer."
)

st.warning(
    "⚠️ Disclaimer: This application is for educational and research purposes only. "
    "It does NOT provide medical diagnosis or medical advice."
)

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("oral_cancer_model.h5")

model = load_model()

# -------------------------
# Image upload
# -------------------------
uploaded_file = st.file_uploader(
    "Upload a mouth image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    st.subheader("Result")

    if prediction > 0.5:
        st.error(
            f"⚠️ Possible abnormal features detected\n\n"
            f"Confidence score: {prediction:.2f}"
        )
    else:
        st.success(
            f"✅ No abnormal features detected\n\n"
            f"Confidence score: {1 - prediction:.2f}"
        )

    st.caption(
        "This result is **not a diagnosis**. "
        "If you have concerns, consult a qualified healthcare professional."
    )
import tensorflow as tf

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("oral_cancer_model.keras")

model = load_model()
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
