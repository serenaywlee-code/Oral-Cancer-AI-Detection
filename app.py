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
