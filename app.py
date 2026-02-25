import streamlit as st
import os
import sys
sys.path.append(os.path.abspath("src"))  # Ensure src is in the path
from src.image_engine import ImageGhoster
from src.text_engine import TextGhoster  # 1. Uncommented
from PIL import Image


# Page Config
st.set_page_config(page_title="Contextual Ghosting", page_icon="👻", layout="wide")

st.title("👻 Contextual Ghosting")
st.markdown("### Hybrid Adversarial Defense Framework")

# Initialize Engines
@st.cache_resource
def get_engines():
    # 2. Return both engines as a tuple
    return ImageGhoster(), TextGhoster()

# 3. Unpack both engines
image_ghoster, text_ghoster = get_engines()

# Sidebar Settings
st.sidebar.header("🛡️ Defense Settings")
eps = st.sidebar.slider("Protection Strength (Epsilon)", 0.001, 0.05, 0.01, format="%.3f")

# Layout
tab1, tab2 = st.tabs(["🖼️ Image Protection", "✍️ Text Protection"])

with tab1:
    st.header("Adversarial Image Ghosting (PGD)")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        temp_path = os.path.join("data/raw", "ui_upload.jpg")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Original Image", use_container_width=True)
            
        with col2:
            with st.spinner("Applying PGD Ghosting..."):
                protected_img = image_ghoster.apply_pgd_protection(temp_path, epsilon=eps)
                st.image(protected_img, caption=f"Protected (Epsilon: {eps})", use_container_width=True)
                
                out_path = "data/processed/ui_output.jpg"
                protected_img.save(out_path)
                with open(out_path, "rb") as file:
                    st.download_button("Download Protected Image", file, "ghosted.jpg", "image/jpeg")

with tab2:
    st.header("Stylometric Text Shuffling")
    user_text = st.text_area("Enter a sentence to protect your writing style:", height=150)
    
    if st.button("Ghost My Text"):
        if user_text:
            with st.spinner("Analyzing linguistic patterns..."):
                # Use the text_ghoster engine we initialized at the top
                protected_text = text_ghoster.shuffle_text(user_text)
                st.subheader("Ghosted Output:")
                st.success(protected_text)
                st.download_button("Download Text", protected_text, file_name="ghosted_text.txt")
        else:
            st.warning("Please type something first!")