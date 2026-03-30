# VisionExtract: Subject Isolation using Image Segmentation

![VisionExtract Banner](docs/images/banner.png)

![Python](https://img.shields.io/badge/Python-3.10-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg)

**VisionExtract** is a high-performance subject isolation tool powered by deep learning. It leverages a customized **U-Net architecture** to perform precise semantic segmentation, extracting the main subject from complex backgrounds with pixel-perfect accuracy.

---

## 🚀 Key Features

- **Automated Subject Isolation**: Instantly separate foreground subjects from backgrounds.
- **Precision Segmentation**: Deep learning-based U-Net model trained on the COCO 2017 dataset.
- **Interactive Web UI**: Real-time subject extraction using Streamlit.
- **Morphological Refining**: Built-in mask cleaning using advanced CV techniques (Opening/Closing) for smoother edges.
- **Batch Processing**: Efficiently process entire directories of images in one go.

---

## 🏗️ Architecture Overview

The core of VisionExtract is a **U-Net** convolutional neural network:

1.  **Encoder (Contracting Path)**: Downsamples the image to capture high-level context and features.
2.  **Bottleneck**: Processes the most compressed representation of the image.
3.  **Decoder (Expanding Path)**: Upsamples back to original resolution while retaining spatial details through **Skip Connections**.
4.  **Post-Processing**: Uses Morphological operations to remove noise and refine subject boundaries.

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/VisionExtract.git
cd VisionExtract
```

### 2. Setup Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📖 Usage Guide

VisionExtract provides an easy-to-use Command Line Interface (CLI) for both training and inference.

### 📍 Subject Isolation (Inference)

To isolate the subject in a single image:
```bash
python src/inference.py --image path/to/your/image.jpg --display
```

To process an entire folder of images:
```bash
python src/inference.py --dir path/to/images --output_dir results/
```

### 🌐 Web Application (UI)

For a more user-friendly experience, you can use the Streamlit interface:
```bash
streamlit run src/app.py
```
This allows you to upload images and download results directly from your browser.

**Common Flags:**
- `--image`: Path to a single image file.
- `--dir`: Path to a directory for batch processing.
- `--checkpoint`: (Optional) Specify a custom `.pth` model file.
- `--display`: Shows the result in a window (for single image mode).

### 🏋️ Model Training

If you want to train the model on your own hardware using the COCO dataset:
```bash
python src/train.py
```
*Checkpoints will be automatically saved in the `checkpoints/` directory.*

---

## 📊 Performance Metrics

The model is evaluated using industry-standard metrics for segmentation:

- **Intersection over Union (IoU)**: Measures the overlap between predicted and ground-truth masks.
- **Dice Coefficient**: High sensitivity to small segmentation errors.
- **Pixel Accuracy**: Overall percentage of correctly classified pixels.

Current benchmarks show high precision on a wide variety of subjects from the COCO test suite.

---

## 📂 Project Structure

```text
VisionExtract/
├── src/                  # Core source code (Model, Train, Inference)
├── data/                 # Dataset storage (COCO 2017)
├── notebooks/            # Experimental JNBs
├── outputs/              # Default inference output directory
├── checkpoints/          # Saved model weights (.pth)
├── docs/                 # Documentation assets
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👤 Author

**Biswajeet Kumar**
- [GitHub](https://github.com/biswajeet111)  <!-- Replace with your actual username -->
- [LinkedIn](www.linkedin.com/in/biswajeet-kumar-a70043362) <!-- Replace with your actual profile -->

---

**Developed as part of the VisionExtract Project.**
