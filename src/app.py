import streamlit as st
import os
import torch
import numpy as np
from PIL import Image
import cv2
import sys
import time
import base64

# Ensure src directory is in path for local imports
sys.path.append(os.path.dirname(__file__))

from inference import VisionExtractPipeline

# Page configuration
st.set_page_config(
    page_title="VisionExtract AI",
    page_icon="🧩",
    layout="wide"
)

# --- Custom Styling & Theme ---
def local_css():
    st.markdown("""
        <style>
        /* Base Theme */
        .main {
            background-color: #050a1a;
            color: #e0e0e0;
        }
        
        /* Glassmorphism Card */
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 16px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 24px;
            margin-bottom: 20px;
        }
        
        /* Gradient Title */
        .gradient-text {
            background: linear-gradient(90deg, #00d2ff, #915aff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 3.5rem;
            margin-bottom: 0px;
        }
        
        .sub-text {
            color: #8892b0;
            font-size: 1.1rem;
            margin-bottom: 30px;
        }
        
        /* Metric Styling */
        .metric-box {
            background: rgba(145, 90, 255, 0.1);
            border-radius: 12px;
            padding: 15px;
            border: 1px solid rgba(145, 90, 255, 0.3);
            text-align: center;
            box-shadow: 0 0 10px rgba(145, 90, 255, 0.1);
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #915aff;
            display: block;
        }
        
        .metric-label {
            font-size: 0.8rem;
            color: #8892b0;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Sidebar Polish */
        .stSidebar {
            background-color: #030712 !important;
        }
        
        /* Button Polish */
        div.stButton > button:first-child {
            background: linear-gradient(90deg, #00d2ff, #915aff);
            color: white;
            border: none;
            padding: 12px 28px;
            border-radius: 8px;
            font-weight: 600;
            transition: 0.3s;
            width: 100%;
        }
        div.stButton > button:first-child:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(145, 90, 255, 0.4);
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

def main():
    # --- Sidebar ---
    st.sidebar.image("docs/images/banner.png", use_container_width=True)
    st.sidebar.markdown("### ⚙️ Engine Control")
    
    checkpoint_dir = "checkpoints"
    available_checkpoints = []
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
        checkpoints.sort(key=lambda x: (x != "best_model.pth", -int(x.split('_')[-1].split('.')[0]) if 'epoch' in x else 0))
        available_checkpoints = checkpoints

    if available_checkpoints:
        selected_checkpoint = st.sidebar.selectbox("Model Version", available_checkpoints)
        model_path = os.path.join(checkpoint_dir, selected_checkpoint)
        st.sidebar.success(f"Loaded: {selected_checkpoint}")
    else:
        st.sidebar.error("⚠️ No models detected.")
        model_path = None

    device = st.sidebar.select_slider("Computation Device", 
        options=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💎 Quality Settings")
    quality_mode = st.sidebar.toggle("Enchanced Quality", value=True, help="Use Guided Filter for sharper edges.")
    
    inf_resolution = st.sidebar.select_slider(
        "Inference resolution", 
        options=[256, 384, 512], 
        value=512,
        help="Higher resolution captures finer details but is slower."
    )
    
    if quality_mode:
        ref_intensity = st.sidebar.slider("Edge Refinement", 0.0, 1.0, 0.8, help="Controls mask sharpness.")
    else:
        ref_intensity = 0.0

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔬 Architecture: ResNet-UNet")
    st.sidebar.caption("High-performance segmentation with pre-trained ResNet34 backbone for precise subject isolation.")

    # --- Header ---
    st.markdown('<h1 class="gradient-text">VisionExtract AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">Intelligent Subject Isolation & Background Extraction</p>', unsafe_allow_html=True)

    # --- Upload Logic ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Drop images here (Multiple supported)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_files:
        st.write(f"📂 **{len(uploaded_files)}** files queued for isolation.")
        
        # Action Bar
        col_btn, col_spacer = st.columns([1, 4])
        process_all = col_btn.button("✨ START EXTRACTION")
        
        if process_all:
            # Initialize Pipeline Once
            pipeline = VisionExtractPipeline(model_path=model_path, device=device, image_size=inf_resolution)
            
            # Progress handling
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Grid Display
            results_container = st.container()
            
            for i, uploaded_file in enumerate(uploaded_files):
                start_time = time.time()
                status_text.text(f"Processing: {uploaded_file.name}...")
                
                # Image Load
                image = Image.open(uploaded_file)
                temp_path = f"temp_{i}.png"
                image.save(temp_path)
                
                try:
                    # Inference
                    isolated_np = pipeline.full_pipeline(
                        temp_path, 
                        save=False, 
                        display=False, 
                        refinement=quality_mode, 
                        refinement_intensity=ref_intensity,
                        custom_size=inf_resolution
                    )
                    inf_time = time.time() - start_time
                    
                    # Display Result Card
                    with results_container:
                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        st.markdown(f"#### 🏷️ Output: {uploaded_file.name}")
                        c1, c2, c3 = st.columns([1, 1, 0.5])
                        
                        with c1:
                            st.image(image, caption="Original", use_container_width=True)
                        with c2:
                            st.image(isolated_np, caption="Isolated", use_container_width=True)
                        with c3:
                            st.markdown(f"""
                                <div class="metric-box">
                                    <span class="metric-value">⏱️ {inf_time:.2f}s</span>
                                    <span class="metric-label">Inference</span>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Download
                            buf = cv2.imencode('.png', cv2.cvtColor(isolated_np, cv2.COLOR_RGB2BGR))[1].tobytes()
                            st.download_button(
                                label="Download PNG",
                                data=buf,
                                file_name=f"isolated_{uploaded_file.name}",
                                mime="image/png",
                                key=f"dl_{i}",
                                use_container_width=True
                            )
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Error on {uploaded_file.name}: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.success("🎉 Batch Processing Complete!")
            st.balloons()
    else:
        # Initial Landing Info
        c_i1, c_i2, c_i3 = st.columns(3)
        with c_i1:
            st.markdown("### 🎯 Accuracy\n0.60+ IoU achieved with Milestone 3 transfer learning.")
        with c_i2:
            st.markdown("### ⚡ Speed\nSub-second inference on RTX 40-series GPUs.")
        with c_i3:
            st.markdown("### 🖼️ Quality\nMorphological mask refining for professional subject isolation.")

if __name__ == "__main__":
    main()
