import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# --------------------------
# Load TFLite model
# --------------------------
@st.cache_resource
def load_model():
    """
    Load TFLite model and allocate tensors.
    """
    interpreter = tflite.Interpreter(model_path="oral_cancer_model_optimized.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# --------------------------
# Prediction function
# --------------------------
def predict(image: Image.Image):
    """
    Preprocess image and run TFLite inference.
    Returns prediction and confidence.
    """
    # Resize and normalize
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0

    # If grayscale, convert to 3 channels
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA -> RGB
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)

    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    confidence = float(output_data[0][0])
    prediction = "Abnormal" if confidence > 0.5 else "Normal"
    return prediction, confidence

# --------------------------
# Streamlit page config
# --------------------------
st.set_page_config(
    page_title="Oral Cancer AI Detection",
    layout="centered",
    page_icon="🦷"
)

# --------------------------
# Title and description
# --------------------------
st.title("🦷 Oral Cancer AI Detection")
st.write(
    "Upload an image of the oral cavity to receive an AI-based screening result. "
    "This tool is for educational purposes only and does not replace professional diagnosis."
)

# --------------------------
# Image uploader
# --------------------------
uploaded_file = st.file_uploader(
    "Upload an oral cavity image (JPG or PNG)", 
    type=["jpg", "jpeg", "png"]
)

# --------------------------
# Display image and analyze
# --------------------------
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing image..."):
            prediction, confidence = predict(image)

        # Show result
        if prediction == "Normal":
            st.success(f"Result: {prediction}")
        else:
            st.warning(f"Result: {prediction}")

        # Show confidence score
        st.info(f"Confidence Score: {confidence*100:.2f}%")

# --------------------------
# Disclaimer
# --------------------------
st.markdown(
    """
    <div style="background-color: rgba(254,202,202,0.45); 
                border-left: 6px solid #dc2626; 
                padding: 16px; 
                border-radius: 10px; 
                color: #b91c1c; 
                margin-top: 20px;">
    <b>Disclaimer:</b><br>
    • This application is a student research project and is not a medical device.<br>
    • Results should not be used for diagnosis or treatment decisions.
    </div>
    """,
    unsafe_allow_html=True
)
