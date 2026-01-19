import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite
import io

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Oral Cancer AI Detection",
    page_icon="🦷",  # replace with "teeth_icon.png" if you have your icon
    layout="centered",
    initial_sidebar_state="auto"
)

# ------------------ CUSTOM CSS ------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Gemini&display=swap');
    
    body {
        background-color: white;
        color: #3275a8;
        font-family: 'Gemini', sans-serif;
    }
    .stButton>button {
        background-color: #3275a8;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1.2em;
    }
    .stApp {
        max-width: 700px;
        margin: auto;
        padding: 2rem;
    }
    h1, h2 {
        color: #3275a8;
    }
    .upload-box {
        border: 2px dashed #3275a8;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ HEADER ------------------
st.title("🦷 Oral Cancer AI Detection")
st.markdown("Upload a mouth image below and our AI will indicate if there may be signs of oral cancer.")
st.markdown("**Note:** This is not an official diagnosis. Please consult a dentist or doctor for medical advice.")

# ------------------ MODEL LOADING ------------------
@st.cache_resource(show_spinner=True)
def load_model():
    # Load TFLite model
    interpreter = tflite.Interpreter(model_path="oral_cancer_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------------------ IMAGE UPLOAD ------------------
uploaded_file = st.file_uploader("Choose a mouth image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image for TFLite model
    img_resized = image.resize((224, 224))  # adjust based on your model input
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)

    # Display result
    if prediction == 1:
        st.error("⚠️ The AI indicates this image may show signs of oral cancer.")
    else:
        st.success("✅ The AI indicates this image appears normal.")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit and TensorFlow Lite")
