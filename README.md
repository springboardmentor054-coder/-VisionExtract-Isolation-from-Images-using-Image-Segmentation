# VisionExtract: Subject Isolation using Image Segmentation

## 📌 Project Overview

VisionExtract is a computer vision project designed to automatically extract the main subject from an image using deep learning-based **semantic segmentation**.

Given an input image, the system generates a new image where:

* The **main subject remains unchanged**
* The **background is converted to black**

This enables automated subject isolation useful for:

* Background removal
* Photo editing automation
* Augmented reality
* Virtual conferencing
* Content creation pipelines

The project uses the **COCO 2017 dataset** and a **U-Net segmentation model** to perform pixel-level subject extraction.

---

# 🎯 Problem Statement

The objective of this project is to build a complete segmentation pipeline capable of:

1. Processing annotated segmentation datasets
2. Generating binary subject masks
3. Training a deep learning segmentation model
4. Evaluating segmentation performance
5. Producing subject-isolated outputs from new images

---

# 📂 Dataset

This project uses the **COCO 2017 Dataset**.

Dataset Source
https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset

Dataset structure used in the project:

```
data/
├── train2017/
└── annotations/
    └── instances_train2017.json
```

* **train2017** → Image dataset
* **instances_train2017.json** → Segmentation annotations

---

# 🏗️ Project Structure

```
VisionExtract/
│
├── data/                 # Dataset storage (not pushed to GitHub)
├── src/                  # Core source code
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── utils.py
│   ├── train.py
│   └── inference.py
│
├── notebooks/            # Development notebooks
├── outputs/              # Prediction results
├── checkpoints/          # Saved model weights
├── requirements.txt
└── README.md
```

---

# 🛠️ Milestone 1 (Week 1)

### Project Initialization & Dataset Exploration

This milestone focused on setting up the project and exploring the segmentation dataset.

### ✅ Completed Tasks

* Project repository initialized
* Virtual environment configured (Python 3.10)
* Required dependencies installed
* COCO 2017 dataset downloaded
* Annotation files extracted and validated
* Dataset explored using **COCO API**
* Visualized sample images and segmentation masks
* Verified dataset structure and annotation alignment
* Initialized Git workflow and feature branches

---

# 🛠️ Milestone 2 (Week 2)

### Data Preprocessing & Initial Model Implementation

This milestone focused on preparing the dataset for training and implementing the initial segmentation model.

### ✅ Completed Tasks

#### Data Preprocessing

* Implemented image preprocessing pipeline
* Resized images and masks to **256 × 256**
* Applied normalization to image tensors
* Converted multi-class segmentation masks into **binary subject masks**
* Ensured alignment between images and masks

#### Dataset Pipeline

* Implemented **PyTorch Dataset class**
* Built **DataLoader pipeline**
* Verified tensor outputs

Example tensor shapes:

```
Image Tensor → (3, 256, 256)
Mask Tensor → (1, 256, 256)
```

---

# 🧠 Segmentation Model

The project implements a **U-Net architecture**, a convolutional neural network designed for pixel-level image segmentation.

### Model Structure

Encoder (Downsampling)

```
256×256 → 128×128 → 64×64 → 32×32
```

Decoder (Upsampling)

```
32×32 → 64×64 → 128×128 → 256×256
```

Skip connections combine encoder and decoder features to preserve spatial information.

---

# ⚙️ Training Setup

Loss Function

```
BCEWithLogitsLoss
```

Optimizer

```
Adam Optimizer
Learning Rate: 0.001
```

Training process includes:

* Forward pass
* Loss computation
* Backpropagation
* Weight updates

---

# 📊 Evaluation Metrics

Segmentation performance is evaluated using:

* **Intersection over Union (IoU)**
* **Dice Coefficient**
* **Pixel-wise Accuracy**

These metrics measure how well predicted masks match ground-truth masks.

---

# 🔬 Prediction Visualization

The model predictions are visualized using:

* Input Image
* Ground Truth Mask
* Predicted Mask

This helps verify subject extraction quality.

---

# 🚀 Remaining Project Timeline

The entire project is planned for **4 weeks** with evaluations every **2 weeks**.

```
Week 1 → Dataset exploration
Week 2 → Preprocessing + initial model
Week 3 → Model training and improvements
Week 4 → Inference pipeline and final evaluation
```

---

# 👨‍💻 Author

VisionExtract Internship Project
Subject Isolation using Deep Learning Segmentation
