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
st.markdown("<h1 style='text-align:center;'>🦷 Oral Health AI Assistant</h1>", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<b>Hi there! 👋</b><br><br>
I’m an AI tool that looks for visual patterns in mouth images.
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📷 Upload a mouth image",
    type=["jpg", "jpeg", "png"]
)
