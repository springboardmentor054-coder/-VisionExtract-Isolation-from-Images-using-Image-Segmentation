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
# VisionExtract – Milestone 3: Prediction & Evaluation

## 📌 Overview
Milestone 3 focuses on generating segmentation predictions using the trained model, evaluating performance, and producing final subject-isolated outputs. This stage completes the VisionExtract pipeline from input image to final extracted subject.

---

## 🎯 Objective
- Generate segmentation masks using trained U-Net model  
- Evaluate model performance using IoU and Dice Score  
- Visualize predictions and isolate subject from background  

---

## 🧠 Pipeline (Milestone 3)
Input Image
↓
Trained U-Net Model
↓
Predicted Mask (probabilities)
↓
Binary Mask (threshold = 0.5)
↓
Mask × Image (pixel-wise)
↓
Isolated Subject Output

---
### Mathematical Representation
Output(x,y) = Image(x,y) × Mask(x,y)
Mask(x,y) ∈ {0,1}
---
---

## ⚙️ Implementation Steps

1. Load trained model and set to evaluation mode  
2. Pass validation images through the model  
3. Apply sigmoid to convert outputs into probabilities  
4. Convert probabilities into binary masks using threshold (0.5)  
5. Compute evaluation metrics (IoU and Dice Score)  
6. Apply mask to original image to isolate subject  
7. Visualize input, ground truth, prediction, and final output  

---

## 📊 Evaluation Metrics

- **IoU (Intersection over Union)**  
  Measures overlap between predicted mask and ground truth  

- **Dice Score**  
  Measures similarity between predicted and actual segmentation  

Higher values indicate better segmentation performance.

---

## 📈 Results

| Metric | Value |
|------|------|
| Mean IoU | ~0.75 |
| Dice Score | ~0.82 |

*(Replace with your actual results)*

---

## 🖼️ Output Visualization

Each prediction includes:
- Input Image  
- Ground Truth Mask  
- Predicted Mask  
- Isolated Subject  

These outputs confirm that the model successfully separates subject from background.

---

## ⚠️ Common Issues

- Incorrect dataset path → FileNotFoundError  
- Mismatch between image and mask names  
- Not using `model.eval()` during prediction  

---

## 🚀 Key Outcome

- Successfully generated segmentation masks  
- Evaluated model performance quantitatively  
- Achieved subject isolation using predicted masks  
- Completed end-to-end VisionExtract pipeline  

---

## 🔮 Future Improvements

- Improve accuracy with advanced models (DeepLabV3+)  
- Optimize threshold selection  
- Deploy real-time segmentation system  

---
# VisionExtract – Milestone 4: Deployment & Application

## 📌 Overview
Milestone 4 focuses on deploying the trained segmentation model into a usable application. This stage enables users to upload an image and automatically extract the main subject using the trained U-Net model.

---

## 🎯 Objective
- Deploy trained segmentation model  
- Accept user input image  
- Generate segmentation mask  
- Isolate subject from background  
- Display final output  

---

## 🧠 Pipeline (Milestone 4)
User Image
↓
Preprocessing (Resize, Normalize)
↓
Load Trained U-Net Model
↓
Prediction (Mask)
↓
Binary Mask (Threshold = 0.5)
↓
Apply Mask to Image
↓
Isolated Subject Output
---

### Mathematical Representation
Output(x,y) = Image(x,y) × Mask(x,y)
Mask(x,y) ∈ {0,1}

---

---

## ⚙️ Implementation

### 1. Load Trained Model
- Load saved weights (`model.pth`)
- Set model to evaluation mode

### 2. Input Image
- User uploads or provides an image  
- Image converted to RGB format  

### 3. Preprocessing
- Resize to 256×256  
- Normalize pixel values  
- Convert to tensor  

### 4. Prediction
- Pass image through model  
- Generate probability mask  
- Convert to binary mask (threshold = 0.5)  

### 5. Subject Isolation
- Apply mask to original image  
- Remove background pixels  

### 6. Output Display
- Show original image  
- Show predicted mask  
- Show isolated subject  

---

## 💻 Application (Streamlit UI)

A simple web interface is built using Streamlit:

### Features:
- Upload image  
- View segmentation result  
- Display isolated subject  

### Run Application:
```bash
streamlit run app.py


---

## ⚙️ Implementation

### 1. Load Trained Model
- Load saved weights (`model.pth`)
- Set model to evaluation mode

### 2. Input Image
- User uploads or provides an image  
- Image converted to RGB format  

### 3. Preprocessing
- Resize to 256×256  
- Normalize pixel values  
- Convert to tensor  

### 4. Prediction
- Pass image through model  
- Generate probability mask  
- Convert to binary mask (threshold = 0.5)  

### 5. Subject Isolation
- Apply mask to original image  
- Remove background pixels  

### 6. Output Display
- Show original image  
- Show predicted mask  
- Show isolated subject  

---

## 💻 Application (Streamlit UI)

A simple web interface is built using Streamlit:

### Features:
- Upload image  
- View segmentation result  
- Display isolated subject  

### Run Application:
```bash
streamlit run app.py

## Author

Dudala Mohana Naga Venkata Sri Lasya
B.Tech CSE (AIML)

---

📌 Conclusion

The VisionExtract project successfully demonstrates an end-to-end image segmentation pipeline for isolating the main subject from images. In Milestone 1, the foundation was established by selecting and understanding a suitable dataset and preparing it for training. Milestone 2 focused on implementing a U-Net model with a pretrained encoder and training it using appropriate loss functions to learn meaningful segmentation patterns. In Milestone 3, the trained model was evaluated using metrics such as IoU and Dice Score, and its effectiveness was verified through visualization of predicted masks and isolated outputs. Finally, in Milestone 4, the model was deployed into a user-friendly application, enabling real-time subject extraction from input images.

Overall, the project highlights the practical application of deep learning in computer vision, covering the complete workflow from data preprocessing to deployment. It demonstrates not only technical implementation but also the ability to build a usable system. This approach can be extended to real-world applications such as photo editing, background removal, surveillance, and content creation.
