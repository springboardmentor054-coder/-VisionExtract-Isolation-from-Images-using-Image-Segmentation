<div align="center">

<img src="docs/images/banner.png" alt="VisionExtract Banner" width="100%">

<br>

# 🌌 VisionExtract: AI-Powered Subject Isolation System

[![Python Version](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

*An advanced machine learning pipeline designed to automatically detect, isolate, and extract primary subjects from images with professional-grade precision.*

[Report Bug](https://github.com/springboardmentor054-coder/-VisionExtract-Isolation-from-Images-using-Image-Segmentation/issues) • [Request Feature](https://github.com/springboardmentor054-coder/-VisionExtract-Isolation-from-Images-using-Image-Segmentation/issues)

</div>

---

## 📖 Table of Contents
- [About the Project](#-about-the-project)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Performance Benchmarks](#-performance-benchmarks)
- [Visual Gallery](#-visual-gallery)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Limitations & Future Scope](#-limitations--future-scope)

---

## 🎯 About the Project

**VisionExtract** is built for digital artists, photographers, and developers who need automated, high-fidelity image cutouts. The system utilizes deep learning to identify the foreground subject and meticulously render the background pixels to absolute black, preserving the subject's natural aspect ratio without distortion.

**Project Objective:**
> To construct a robust, edge-aware ML model capable of automated subject extraction, yielding a clean, production-ready image where only the main subject remains visible.

---

## ✨ Key Features

* 🧠 **Deep Learning Driven:** Powered by a robust ResNet34-UNet architecture via Transfer Learning.
* ⚖️ **Aspect-Ratio Preservation:** Utilizes `LongestMaxSize` transformations to prevent spatial distortion during preprocessing.
* ⚡ **GPU Accelerated:** Built-in CUDA support with Automatic Mixed Precision (AMP) for lightning-fast sub-second inference.
* 🎨 **Interactive Dashboard:** Features a premium **Streamlit** frontend for real-time visualization and background replacement (e.g., Studio, Nature, Office).
* 🔪 **High-Fidelity Edges:** Advanced matting logic for smooth, anti-aliased transitions between subject and background.

---

## 🛠️ System Architecture

Our tech stack is strictly curated for high performance in computer vision tasks:

| Component | Technology Used |
| :--- | :--- |
| **Core Architecture** | PyTorch, ResNet34 (Backbone), UNet |
| **Image Processing** | OpenCV, Albumentations |
| **Frontend/UI** | Streamlit |
| **Optimization** | IoU-based Checkpointing, Adam Optimizer |

---

## 📊 Performance Benchmarks

The model was rigorously trained over **110 epochs**, including a specialized 10-epoch Refinement Phase at a reduced learning rate (`5e-5`), yielding the following metrics:

| Evaluation Metric | Score / Achievement |
| :--- | :--- |
| **Mean Intersection over Union (IoU)** | **0.64+** |
| **Dice Coefficient** | **0.78+** |
| **Average Inference Time** | **~0.15 seconds** (GPU) |

<details>
<summary><b>Click to view Model Comparisons</b></summary>

| Architecture | Baseline IoU | VisionExtract Optimized IoU |
| :--- | :--- | :--- |
| Standard UNet | 0.47 | - |
| **ResNet34-UNet** | - | **0.62 - 0.64+** |

</details>

---

## 🖼️ Visual Gallery

Below are real output samples demonstrating the model's ability to cleanly extract complex subjects from diverse backgrounds.

| Original Input Image | Isolated Subject (VisionExtract) |
| :---: | :---: |
| <img src="outputs/Input1.jpg" width="300" alt="Input 1"> | <img src="outputs/Output1.jpg" width="300" alt="Output 1"> |
| <img src="outputs/Input2.jpg" width="300" alt="Input 2"> | <img src="outputs/Output2.jpg" width="300" alt="Output 2"> |
| <img src="outputs/Input3.jpg" width="300" alt="Input 3"> | <img src="outputs/Output3.jpg" width="300" alt="Output 3"> |

---

## 🚀 Getting Started

Follow these instructions to get a local copy up and running for development and testing.

### Prerequisites
* Python 3.10 or higher
* Git
* (Optional but recommended) NVIDIA GPU with CUDA toolkit installed

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/springboardmentor054-coder/-VisionExtract-Isolation-from-Images-using-Image-Segmentation.git](https://github.com/springboardmentor054-coder/-VisionExtract-Isolation-from-Images-using-Image-Segmentation.git)
   cd -VisionExtract-Isolation-from-Images-using-Image-Segmentation
