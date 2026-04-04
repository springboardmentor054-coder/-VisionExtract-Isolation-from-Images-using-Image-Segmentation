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
        # Sort to put best_model.pth first, then epoch-descending
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
    st.sidebar.markdown("### 🖼️ Background Style")
    bg_options = {
        "Deep Black": "black",
        "Modern Office": "docs/images/backgrounds/office.png",
        "Lush Nature": "docs/images/backgrounds/nature.png",
        "Photo Studio": "docs/images/backgrounds/studio.png",
        "Soft Blur": "blur"
    }
    selected_bg = st.sidebar.selectbox("Virtual Background", list(bg_options.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔬 Architecture: ResNet-UNet")
    st.sidebar.caption("High-performance segmentation with pre-trained ResNet34 backbone for precise subject isolation.")

    # --- Header ---
    st.markdown('<h1 class="gradient-text">VisionExtract AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">Intelligent Subject Isolation & Background Extraction</p>', unsafe_allow_html=True)

    # --- Tabs ---
    tab_extract, tab_tech = st.tabs(["✨ Extraction Engine", "📊 Technical Dashboard"])

    with tab_extract:
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
                # Initialize Pipeline Once (Standard 256 mode)
                pipeline = VisionExtractPipeline(model_path=model_path, device=device, image_size=256)
                
                def apply_background(img_np, mask_np, bg_type):
                    h, w = img_np.shape[:2]
                    if bg_type == "black":
                        return (img_np * mask_np[:, :, None]).astype(np.uint8)
                    elif bg_type == "blur":
                        background = cv2.GaussianBlur(img_np, (21, 21), 0)
                    else:
                        if os.path.exists(bg_options[bg_type]):
                            background = cv2.imread(bg_options[bg_type])
                            background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
                            background = cv2.resize(background, (w, h))
                        else:
                            return (img_np * mask_np[:, :, None]).astype(np.uint8)
                    
                    # Alpha Blending with soft-mask for smooth matting
                    mask_3d = mask_np[:, :, None]
                    blended = (img_np * mask_3d + background * (1 - mask_3d)).astype(np.uint8)
                    return blended
    
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
                        # Standard Pipeline (No aggressive thinning)
                        isolated_black, soft_mask = pipeline.full_pipeline(
                            temp_path, 
                            save=False, 
                            display=False
                        )
                        
                        # Apply selected background
                        final_output = apply_background(np.array(image), soft_mask, selected_bg)
                        
                        inf_time = time.time() - start_time
                        
                        # Display Result Card
                        with results_container:
                            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                            st.markdown(f"#### 🏷️ Output: {uploaded_file.name}")
                            c1, c2, c3 = st.columns([1, 1, 0.5])
                            
                            with c1:
                                st.image(image, caption="Original", use_container_width=True)
                            with c2:
                                st.image(final_output, caption=f"Result ({selected_bg})", use_container_width=True)
                            with c3:
                                st.markdown(f"""
                                    <div class="metric-box">
                                        <span class="metric-value">⏱️ {inf_time:.2f}s</span>
                                        <span class="metric-label">Inference</span>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown("<br>", unsafe_allow_html=True)
                                
                                # Download
                                buf = cv2.imencode('.png', cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR))[1].tobytes()
                                st.download_button(
                                    label="Download PNG",
                                    data=buf,
                                    file_name=f"visionextract_{uploaded_file.name}",
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

    # --- Technical Dashboard ---
    with tab_tech:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Model Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg. IoU", "0.621", "+0.02")
        m2.metric("Dice Score", "0.756", "+0.01")
        m3.metric("Pixel Accuracy", "90.2%", "+0.5%")
        m4.metric("Inf. Speed", "0.15s", "-0.05s")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🏗️ Architecture Overview")
        st.info("**Encoder:** ResNet34 (ImageNet Pre-trained)\n\n**Decoder:** Symmetric UNet with skip-connections and Bilinear Upsampling.\n\n**Pipeline:** Standardized Aspect-Ratio Aware Inference (256px Base).")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🚀 Showcase Readiness")
        st.success("- [x] Robust Multi-image Batch Processing\n- [x] Standard Linear Up-scaling Matting\n- [x] Dynamic Virtual Background Replacement\n- [x] Optimized Performance for Final Demo")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
