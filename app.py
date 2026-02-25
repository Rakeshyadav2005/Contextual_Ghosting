import streamlit as st
from src.image_engine import ImageGhoster
from PIL import Image
import os

# Page Config
st.set_page_config(page_title="Contextual Ghosting", page_icon="👻", layout="wide")

st.title("👻 Contextual Ghosting")
st.markdown("### Hybrid Adversarial Defense Framework")

# Initialize Engine
@st.cache_resource
def get_ghoster():
    return ImageGhoster()

ghoster = get_ghoster()

# Layout
tab1, tab2 = st.tabs(["🖼️ Image Protection", "✍️ Text Protection"])

with tab1:
    st.header("Adversarial Image Ghosting")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        # Save temp file
        temp_path = os.path.join("data/raw", "ui_upload.jpg")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Original Image", use_container_width=True)
            
        with col2:
            with st.spinner("Applying Ghosting..."):
                protected_img = ghoster.apply_protection(temp_path)
                st.image(protected_img, caption="Protected (Ghosted) Image", use_container_width=True)
                
                # Save and Download
                out_path = "data/processed/ui_output.jpg"
                protected_img.save(out_path)
                with open(out_path, "rb") as file:
                    st.download_button("Download Protected Image", file, "ghosted.jpg", "image/jpeg")

with tab2:
    st.header("Stylometric Text Shuffling")
    st.info("Phase 2 Text Engine integration coming soon...")