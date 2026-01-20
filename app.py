import os
import tensorflow as tf
import streamlit as st
import gdown

@st.cache_resource
def load_model():
    model_path = "oral_cancer_model_optimized.tflite"

    # OPTIONAL: Google Drive download (recommended for big files)
    FILE_ID = "PASTE_YOUR_GOOGLE_DRIVE_FILE_ID_HERE"

    if not os.path.exists(model_path):
        st.info("⬇️ Downloading AI model...")
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            model_path,
            quiet=False
        )

    if not os.path.exists(model_path):
        st.error("❌ Model file not found.")
        st.stop()

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter
