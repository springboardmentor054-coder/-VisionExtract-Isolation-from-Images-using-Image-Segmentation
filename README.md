# 🚀 VisionExtract: Subject Isolation using Image Segmentation

---

## 📌 Project Overview

**VisionExtract** is a deep learning-based image segmentation project aimed at isolating the main subject (e.g., a person) from an image.

The system takes an input image and generates an output where:

* The subject is preserved
* The background is completely removed (set to black)

This is achieved using **semantic segmentation techniques** and deep neural networks.

---

## 🎯 Project Objectives

* Load and explore a segmentation dataset (COCO)
* Generate binary masks for the main subject
* Build a complete preprocessing pipeline
* Train segmentation models (UNet-based)
* Improve performance using advanced architectures
* Deploy an inference pipeline for real-world usage

---

## 📂 Dataset

* **Dataset**: COCO 2017
* **Annotations**: `instances_train2017.json`
* **Category Used**: `person`
* **Tool Used**: `pycocotools`

### ✔ Mask Creation

* Multi-object annotations converted into **binary masks**
* Pixel values:

  * `1 → Subject`
  * `0 → Background`

---

# 🧩 Milestone 1

## ⚙️ Week 1: Dataset Acquisition & Exploration

* Loaded COCO dataset using COCO API
* Extracted:

  * Category IDs
  * Image IDs
* Generated segmentation masks using `annToMask()`
* Visualized:

  * Original Image
  * Binary Mask
  * Subject-Isolated Image

### ✅ Outcome

Understanding of dataset structure and mask generation process.

---

## ⚙️ Week 2: Data Preprocessing Pipeline

### ✔ Implemented Custom Dataset Class

* Built using PyTorch `Dataset`
* Dynamically loads image-mask pairs

### ✔ Preprocessing Steps

* Resize images to **256 × 256**
* Normalize images using ImageNet statistics
* Convert masks to binary format
* Use **nearest-neighbor interpolation** for masks

### ✔ Data Splitting

* 80% Training
* 20% Validation

### ✔ DataLoader

* Batch loading implemented
* Shuffle enabled for training

### ✅ Outcome

A fully functional and scalable data pipeline for segmentation.

---

# 🧠 Milestone 2

## ⚙️ Week 3: Initial Model Training

### ✔ Model Used

* Custom **U-Net architecture**

### ✔ Training Setup

* Loss Function: `BCEWithLogitsLoss`
* Optimizer: Adam
* Input Size: 256 × 256

### ✔ Metrics

* **Intersection over Union (IoU)** used for evaluation

### ✔ Training Process

* Forward pass → Loss computation → Backpropagation → Optimization
* Training and validation loops implemented

---

## ⚙️ Week 4: Prediction & Fine-Tuning

### ✔ Validation Analysis

* Generated predictions on validation data
* Compared:

  * Ground truth masks
  * Predicted masks

### ✔ Visualization

* Input Image
* Ground Truth Mask
* Predicted Mask

### ✔ Hyperparameter Experiment

* Learning Rate comparison:

  * `1e-3` vs `1e-4`
* Observed performance differences using:

  * Validation Loss
  * IoU

### ✅ Outcome

Baseline segmentation model successfully trained and evaluated.

---

# 🚀 Milestone 3

## ⚙️ Week 5: Data Improvement & Model Enhancement

### ✔ Advanced Data Augmentation

To improve generalization:

* Horizontal Flip
* Vertical Flip
* Rotation
* Brightness & Contrast Adjustment
* Gaussian Blur

### ✔ Improved Model Architecture

* Replaced baseline U-Net with **ResNet34-based U-Net**
* Used **pretrained ImageNet weights**

### 🔥 Benefits:

* Better feature extraction
* Faster convergence
* Improved segmentation quality

---

### ✔ Model Comparison

| Model        | Description          | Performance |
| ------------ | -------------------- | ----------- |
| Basic U-Net  | Trained from scratch | Moderate    |
| ResNet U-Net | Pretrained encoder   | High        |

### ✅ Outcome

Improved model produces:

* Cleaner masks
* Better edge detection
* Strong generalization

---

## ⚙️ Week 6: Inference Pipeline

### ✔ Deployment Pipeline

Developed a complete inference system:

```
Input Image → Preprocessing → Model → Mask → Output Image
```

### ✔ Features

* Accepts unseen images
* Generates segmentation mask
* Applies mask to isolate subject
* Converts background to black

---

### ✔ Output

* Subject preserved
* Background removed
* Works on real-world images

---

### ✔ Robustness Testing

* Tested on images outside dataset
* Model shows good generalization

---

## 🏁 Final Outcome

* Complete end-to-end segmentation system
* From dataset → training → inference
* Ready for real-world applications

---

## 🌍 Applications

* Background removal
* Photo editing tools
* Virtual conferencing
* Augmented reality
* Digital content creation

---

## 🧠 Key Learnings

* Semantic segmentation fundamentals
* Data preprocessing techniques
* Model training & evaluation
* Importance of augmentation
* Power of pretrained architectures
* Deployment of ML pipelines

---

## 🔮 Future Scope

* Build web interface (Streamlit)
* Real-time video segmentation
* Multi-class segmentation
* Edge device deployment

---

## 👨‍💻 Author

**Dinesh Chowdary Vattikunta**

---

## ⭐ Conclusion

VisionExtract successfully demonstrates how deep learning can be used to isolate subjects from images with high accuracy.
The improvements in Milestone 3 significantly enhanced both performance and usability of the system.

---
