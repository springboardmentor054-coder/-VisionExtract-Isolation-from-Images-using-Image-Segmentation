import streamlit as st
import os
import torch
import numpy as np
from PIL import Image
import cv2
import sys
import time

# Ensure src directory is in path for local imports
sys.path.append(os.path.dirname(__file__))

from inference import VisionExtractPipeline

# Page configuration
st.set_page_config(
    page_title="VisionExtract - Subject Isolation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border: none;
    }
    .upload-text {
        color: #ccd6f6;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .title-text {
        background: linear-gradient(90deg, #ff4b4b, #ff8a8a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0px;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Sidebar
    st.sidebar.title("Configuration")
    checkpoint_dir = "checkpoints"
    available_checkpoints = []
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
        # Sort such that best_model.pth is at the top, then others by epoch descending
        checkpoints.sort(key=lambda x: (x != "best_model.pth", -int(x.split('_')[-1].split('.')[0]) if 'epoch' in x else 0))
        available_checkpoints = checkpoints

    if available_checkpoints:
        selected_checkpoint = st.sidebar.selectbox("Select Model Checkpoint", available_checkpoints)
        model_path = os.path.join(checkpoint_dir, selected_checkpoint)
    else:
        st.sidebar.warning("No checkpoints found in 'checkpoints/' directory.")
        model_path = None

    device = st.sidebar.radio("Device", ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **VisionExtract** uses a customized U-Net architecture to isolate the main subject from any image.
    
    **Instructions:**
    1. Select a model checkpoint (if available).
    2. Upload an image (JPG/PNG).
    3. Click 'Process Image' to see the result.
    4. Download the isolated subject.
    """)

    # Main Content
    st.markdown('<h1 class="title-text">VisionExtract</h1>', unsafe_allow_html=True)
    st.markdown('<p class="upload-text">Premium Subject Isolation using Deep Learning</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        # Original Image
        image = Image.open(uploaded_file)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        # Process Button
        if st.button("Extract Subject"):
            with st.spinner("Processing... isolatng subject..."):
                # Initialize Pipeline
                pipeline = VisionExtractPipeline(model_path=model_path, device=device)
                
                # Save uploaded file temporarily for pipeline
                temp_path = "temp_upload.png"
                image.save(temp_path)
                
                try:
                    # Run Pipeline
                    isolated_np = pipeline.full_pipeline(temp_path, save=False, display=False)
                    isolated_image = Image.fromarray(isolated_np)
                    
                    with col2:
                        st.subheader("Isolated Subject")
                        st.image(isolated_image, use_container_width=True)
                        
                        # Download button
                        btn = st.download_button(
                            label="Download Result",
                            data=cv2.imencode('.png', cv2.cvtColor(isolated_np, cv2.COLOR_RGB2BGR))[1].tobytes(),
                            file_name=f"isolated_{uploaded_file.name.split('.')[0]}.png",
                            mime="image/png"
                        )
                    
                    st.success("Successfully isolated the subject!")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

if __name__ == "__main__":
    main()
