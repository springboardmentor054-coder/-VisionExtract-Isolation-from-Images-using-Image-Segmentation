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
│   ├── mask_visualization.ipynb
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
# VisionExtract – Milestone 2 Notes

## Objective
The goal of Milestone 2 is to implement and train a deep learning segmentation model that can identify and isolate the main subject in an image.

---

## Model Architecture

The model used is **U-Net with a ResNet34 encoder**.

Why U-Net?

- Designed for image segmentation
- Works well with limited datasets
- Captures both spatial and contextual features

Encoder: ResNet34 (pretrained on ImageNet)

Input:
RGB images (3 channels)

Output:
Binary mask (subject vs background)

---

## Loss Functions

Two loss functions are used:

### Binary Cross Entropy Loss
Measures pixel-wise difference between predicted mask and ground truth mask.

### Dice Loss
Measures overlap between predicted segmentation and actual mask.

Combining both losses improves segmentation accuracy.

---

## Training Process

Training is performed for **5 epochs**.

Steps during training:

1. Load image and mask batch
2. Pass images through U-Net
3. Generate predicted masks
4. Calculate loss
5. Perform backpropagation
6. Update model weights

---

## Evaluation Metric

The model is evaluated using **Intersection over Union (IoU)**.

Formula:

IoU = Intersection / Union

Where:

Intersection = overlap between predicted and actual mask

Union = total combined area of predicted and actual mask

Higher IoU indicates better segmentation accuracy.

---

## Validation

After each epoch:

- The model is switched to evaluation mode
- Validation images are passed through the model
- IoU score is calculated
- Performance metrics are recorded

---

## Output

The model generates:

- Predicted segmentation mask
- Isolated subject image

The output demonstrates the model’s ability to separate subject pixels from the background.

## Model Training

This stage focuses on implementing and training a U-Net based segmentation model.

Steps:

1. Load training and validation datasets
2. Build U-Net with ResNet34 encoder
3. Train model using BCE + Dice loss
4. Evaluate using IoU metric
5. Visualize predicted masks

The model learns to generate binary segmentation masks that isolate the subject from the background.

---
## Author

Dudala Mohana Naga Venkata Sri Lasya
B.Tech CSE (AIML)

---

## Conclusion

This project demonstrates the complete pipeline for subject isolation using semantic segmentation and prepares the foundation for building advanced background removal systems.
