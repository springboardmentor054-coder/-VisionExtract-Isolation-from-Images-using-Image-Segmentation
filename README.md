# VisionExtract: Subject Isolation from Images using Semantic Segmentation

## Overview

VisionExtract is a deep learning project that automatically isolates the main subject from an image by removing the background using semantic segmentation.

The system takes an input image and generates an output image where only the subject is visible and the background is completely black.

This project uses the COCO 2017 dataset and mask annotations to understand and implement pixel-wise segmentation.

---

## Project Objective

The main objectives of this project are:

* Understand semantic segmentation concepts
* Work with COCO dataset and annotation files
* Generate segmentation masks from annotations
* Perform subject isolation using binary masks
* Prepare dataset for training segmentation models
* Build a complete pipeline for background removal

---

## Example Workflow

Image → Annotation → Mask → Binary Mask → Isolated Subject

---

## Dataset

This project uses the COCO 2017 dataset.

Download from:

https://cocodataset.org/#download

Required files:

* train2017.zip
* annotations_trainval2017.zip

---

## Folder Structure

```
VisionExtract/
│
├── data/
│   ├── train2017/
│   ├── annotations/
│
├── notebooks/
│   ├── VisionExtract_Supervisely.ipynb
│
├── outputs/
│
├── requirements.txt
│
└── README.md
```

---

## Installation

Install required libraries:

```
pip install numpy matplotlib pillow pycocotools torch torchvision
```

---

## Implementation Steps

### 1. Load COCO Dataset

Load annotation file using pycocotools.

### 2. Explore Categories

Identify object categories such as person, car, dog, etc.

### 3. Load Images

Load images containing the target object (person).

### 4. Generate Masks

Convert annotations into pixel-wise masks.

### 5. Create Binary Mask

Convert mask into binary format:

* 1 → Subject
* 0 → Background

### 6. Perform Subject Isolation

Multiply original image with binary mask to remove background.

---

## Sample Output

Original Image → Mask → Isolated Subject

---

## Technologies Used

* Python
* NumPy
* Matplotlib
* Pillow
* Pycocotools
* PyTorch

---

## Key Concepts Learned

* Semantic Segmentation
* COCO Dataset Structure
* Annotation Processing
* Mask Generation
* Image Processing

---

## Future Work

* Train U-Net segmentation model
* Improve segmentation accuracy
* Build web application interface
* Deploy model

---

## Applications

* Background removal
* Photo editing
* Virtual background
* Augmented reality
* E-commerce

---

## Author

Dudala Mohana Naga Venkata Sri Lasya
B.Tech CSE (AIML)

---

## Conclusion

This project demonstrates the complete pipeline for subject isolation using semantic segmentation and prepares the foundation for building advanced background removal systems.
