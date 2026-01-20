import os
import tensorflow as tf
import streamlit as st

@st.cache_resource
def load_model():
    model_path = "oral_cancer_model_optimized.tflite"

    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Make sure oral_cancer_model_optimized.tflite is uploaded.")
        st.stop()

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter
