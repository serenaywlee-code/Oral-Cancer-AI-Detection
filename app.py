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
html, body, .main {
    background-color: #dff0fb !important;
}

.block-container {
    padding-top: 2rem;
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

/* Main white card */
.card {
    background: white;
    padding: 36px;
    border-radius: 26px;
    box-shadow: 0px 12px 30px rgba(0,0,0,0.08);
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
Upload an image of the oral cavity to receive an AI-based screening result.<br>
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

# ---------------- MAIN CARD (ONLY ONE) ----------------
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
import { useState, useRef } from 'react';
import { Upload, FileImage, AlertCircle, CheckCircle2 } from 'lucide-react';

export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<'normal' | 'abnormal' | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (file: File | null) => {
    if (file && (file.type === 'image/jpeg' || file.type === 'image/png')) {
      if (file.size <= 200 * 1024 * 1024) { // 200MB limit
        setSelectedFile(file);
        setResult(null);
        // Simulate AI analysis
        setIsAnalyzing(true);
        setTimeout(() => {
          setIsAnalyzing(false);
          // Mock result - randomly show normal or abnormal for demo
          setResult(Math.random() > 0.5 ? 'normal' : 'abnormal');
        }, 2500);
      } else {
        alert('File size exceeds 200MB limit');
      }
    } else {
      alert('Please upload a JPG or PNG file');
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFileSelect(file);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    handleFileSelect(file);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="max-w-3xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <span className="text-5xl">🦷</span>
            <h1 className="text-4xl font-bold text-gray-900">
              Oral Cancer AI Detection
            </h1>
          </div>
          <div className="text-gray-700 text-lg space-y-1" style={{ fontFamily: 'Lora, serif' }}>
            <p>Upload an image of the oral cavity to receive an AI-based screening result.</p>
            <p>This tool is for educational purposes only and does not replace professional diagnosis.</p>
          </div>
        </div>

        {/* Upload Card */}
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">
            Upload an oral cavity image (JPG or PNG)
          </h2>

          {/* File Upload Area */}
          <div
            className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${
              isDragging
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-300 bg-gray-50 hover:border-gray-400'
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png"
              onChange={handleFileInputChange}
              className="hidden"
            />

            <div className="flex flex-col items-center gap-4">
              {selectedFile ? (
                <>
                  <FileImage className="w-16 h-16 text-green-600" />
                  <div className="text-gray-900 font-medium">{selectedFile.name}</div>
                  <div className="text-sm text-gray-500">
                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                  </div>
                </>
              ) : (
                <>
                  <Upload className="w-16 h-16 text-gray-400" />
                  <div className="text-gray-600 font-medium">No file chosen</div>
                  <div className="text-gray-500">Drag and drop file here</div>
                </>
              )}

              <button
                onClick={() => fileInputRef.current?.click()}
                className="mt-2 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
              >
                Choose File
              </button>

              <div className="text-sm text-gray-500 mt-2">
                Limit 200MB per file • JPG, JPEG, PNG
              </div>
            </div>
          </div>

          {/* Analysis Result */}
          {isAnalyzing && (
            <div className="mt-6 p-6 bg-blue-50 rounded-xl border border-blue-200">
              <div className="flex items-center gap-3">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                <span className="text-blue-900 font-medium">Analyzing image...</span>
              </div>
            </div>
          )}

          {result === 'normal' && !isAnalyzing && (
            <div className="mt-6 p-6 bg-green-50 rounded-xl border border-green-200">
              <div className="flex items-start gap-3">
                <CheckCircle2 className="w-6 h-6 text-green-600 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-green-900 mb-1">Normal Result</h3>
                  <p className="text-green-800">
                    No abnormalities detected in the uploaded image. However, please consult 
                    a healthcare professional for accurate diagnosis.
                  </p>
                </div>
              </div>
            </div>
          )}

          {result === 'abnormal' && !isAnalyzing && (
            <div className="mt-6 p-6 bg-amber-50 rounded-xl border border-amber-200">
              <div className="flex items-start gap-3">
                <AlertCircle className="w-6 h-6 text-amber-600 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-amber-900 mb-1">Abnormality Detected</h3>
                  <p className="text-amber-800">
                    The AI has detected potential abnormalities. Please consult a healthcare 
                    professional immediately for proper examination and diagnosis.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Disclaimer */}
        <div className="bg-red-50 border-l-4 border-red-500 rounded-lg p-6">
          <h3 className="font-bold text-red-900 mb-3">Disclaimer:</h3>
          <ul className="space-y-2 text-red-800">
            <li className="flex items-start gap-2">
              <span className="text-red-500 mt-1">•</span>
              <span>This application is a student research project and is not a medical device.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-red-500 mt-1">•</span>
              <span>Results should not be used for diagnosis or treatment decisions.</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
