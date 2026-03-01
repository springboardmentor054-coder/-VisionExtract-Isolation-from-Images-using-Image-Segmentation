# VisionExtract: Subject Isolation using Image Segmentation

## 📌 Project Overview
VisionExtract is an image segmentation project focused on isolating the main subject (e.g., person) from an image using pixel-level mask annotations.

The project uses the COCO dataset and builds a complete preprocessing pipeline to prepare data for deep learning-based segmentation models.

---

## 🎯 Project Objectives
- Load and explore a segmentation dataset (COCO).
- Generate binary masks for the main subject.
- Build a preprocessing pipeline.
- Prepare training and validation datasets.
- Enable future model training (e.g., UNet).

---

## 📂 Dataset
- Dataset: COCO 2017
- Annotations: `instances_train2017.json`
- Category used: `person`
- Binary mask creation using `pycocotools`

---

## ⚙️ Week 1: Dataset Acquisition & Exploration
- Loaded COCO dataset.
- Explored category IDs and image IDs.
- Generated segmentation masks using `annToMask`.
- Visualized:
  - Original image
  - Binary mask
  - Isolated subject

---

## ⚙️ Week 2: Data Preprocessing Pipeline
Implemented a PyTorch Dataset class:

### ✔ Features:
- Loads image and corresponding mask
- Converts multi-class masks into binary masks
- Resizes images to 256×256
- Resizes masks using nearest-neighbor interpolation
- Converts to PyTorch tensors
- Splits dataset into 80% training and 20% validation

---

## 🧱 Project Structure
