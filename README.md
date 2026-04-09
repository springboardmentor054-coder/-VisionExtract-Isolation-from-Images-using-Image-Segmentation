# VisionExtract: Isolation from Images using Image Segmentation

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

## Project Statement

The goal of this project is to build a machine learning model capable of automatically extracting the main subject from an image. For any given input picture, the model should output a new image in which only the subject is visible and everything else is rendered completely black. This subject isolation process can be used for automation in photography, digital art, augmented reality, virtual conferencing, and background replacement applications.

## Use Cases

The project addresses one primary use case:

### Automated Subject Isolation
- **Description**: Automatically detect and extract the main subject from any image. The output will be an image where only the subject is displayed as in the original photo, while the rest of the pixels are set to black. This replicates the "cutout" functionality required in many media editing pipelines.

## Learning Outcomes

By the end of this project, you will:

- Understand the principles of semantic segmentation using deep learning models for pixel-wise image tasks.
- Learn and implement data preprocessing strategies for segmentation, including managing mask annotations and input normalization.
- Train, validate, and evaluate an image segmentation model for the specific application of subject isolation.
- Deploy and demonstrate a solution that processes user-uploaded images and returns the desired isolated-subject results.
- Prepare detailed documentation and a presentation on your data pipeline, modeling approaches, and results.

## Dataset

This project uses the [COCO 2017 Dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) from Kaggle, which provides images with instance segmentation annotations suitable for training subject isolation models.

- **Key Features**: High-quality images with detailed mask annotations for multiple objects. Masks can be processed to focus on the primary subject.
- **Download Instructions**: Sign up on Kaggle, download the dataset, and place it in a `data/` directory (see setup below).

## Model Architecture

The model employs a U-Net architecture with ResNet34 backbone for semantic segmentation to generate pixel-wise binary masks. During inference, the mask is applied to retain original pixels for the subject and set background pixels to black. Additional architectures like LinkNet were explored for comparison.

### High-Level Pipeline Diagram

```
User Upload → Image Preprocessing (Resize, Normalize) → U-Net Model → Binary Mask → 
Morphological Post-Processing → Apply Mask to Image → Isolated Subject Output
```

- **Input**: RGB image (resized to 256x256 during training/inference).
- **Output**: Isolated subject image with black background.
- **Backbone**: ResNet34 for feature extraction.
- **Deployment**: Local Flask web application with HTML/CSS UI.

## Modules Implemented

### 1. Data Preprocessing and Feature Engineering
- Handle raw image and mask data preparation for consistency.
- Apply data augmentation (resize, horizontal flip, elastic transform, grid distortion, brightness/contrast).
- Convert multi-class COCO masks to binary subject-background format.
- Ensure alignment between images and masks for accurate training.

### 2. Building the Segmentation Model
- Implemented U-Net with ResNet34 backbone for pixel-wise binary segmentation.
- Explored LinkNet architecture for comparison.
- During inference: Retain input pixels for subject; replace background with black.

### 3. Evaluation and Fine-Tuning
- Quantitative metrics: IoU, Dice Coefficient, Pixel-wise Accuracy.
- Qualitative review: Inspect sample output images for visual fidelity.
- Fine-tuned with hyperparameters, augmentations, and post-processing (morphological operations).

### 4. Web Application and Deployment
- Flask-based web app for image upload and real-time inference.
- Beautiful UI with Tailwind CSS for user interaction.
- Automated pipeline from upload to result display.

## Implementation Milestones

The project is structured over 8 weeks with weekly deliverables and screenshots of outputs.

### Milestone 1: Project Initialization and Data Setup ✅ COMPLETED
- **Week 1: Project Initialization and Dataset Acquisition**
  - Defined objectives and acquired/inspected COCO 2017 dataset.
  - Explored structure; viewed example images and masks (COCO.ipynb).
- **Week 2: Data Preprocessing and Validation**
  - Built preprocessing pipeline (resize, normalization, augmentation).
  - Ensured image-mask alignment; converted multi-class to binary masks.

### Milestone 2: Initial Model Training ✅ CODE COMPLETE
- **Week 3: Initial Model Training**
  - Implemented U-Net segmentation network with ResNet34 backbone.
  - Training pipeline with loss monitoring and metrics (milestone2.ipynb).
  - Early prediction visualization.
- **Week 4: Predictions and Fine-Tuning**
  - Validation predictions; comparison with ground-truth.
  - Hyperparameter adjustments and data augmentation.

### Milestone 3: Improvements and Inference ✅ CODE COMPLETE
- **Week 5: Improve Data and Experiment with Architectures**
  - Refined preprocessing with advanced augmentations.
  - Explored LinkNet architecture; documented comparisons (milestone3.ipynb).
- **Week 6: Inference**
  - Deployed model for inference on new images.
  - Automated pipeline; tested on unseen data.

### Milestone 4: Full Pipeline and Demo ✅ COMPLETED
- **Week 7: Full Pipeline and User Interface**
  - Built Flask web app for image upload and result display (app.py, templates/index.html).
  - Integrated preprocessing, model inference, and output generation.
- **Week 8: Documentation, Presentation, and Demo**
  - Compiled comprehensive documentation (this README).
  - Prepared presentation materials with screenshots.
  - Ready for live demonstration.

## Setup and Installation

1. **Prerequisites**: Python 3.8+, pip, and a GPU (optional but recommended for training).
2. **Clone/Download**: Ensure all project files are in the workspace.
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Dataset Setup**:
   - Download COCO 2017 dataset from Kaggle.
   - Extract to a `data/` folder in the project root.
   - Ensure `data/train2017/`, `data/val2017/`, and `data/annotations/` exist.
5. **Model Training** (if needed):
   - Run cells in `milestone2.ipynb` for initial training.
   - Run cells in `milestone3.ipynb` for refined training and architecture comparison.
   - Trained model will be saved as `Model/best_unet_model.pth`.

## Usage

### Web Application
1. Ensure the trained model `Model/best_unet_model.pth` exists.
2. Run the Flask app:
   ```bash
   python app.py
   ```
3. Open `http://localhost:5000/` in your browser.
4. Upload an image to see the isolated subject result.

### Notebooks
- `COCO.ipynb`: Dataset exploration and preprocessing.
- `milestone2.ipynb`: Initial model training and validation.
- `milestone3.ipynb`: Advanced training, architecture comparison, and inference.

## Results and Evaluation

### Quantitative Results
After training and evaluating three architectures (U-Net, LinkNet, DeepLabV3+), the following mean IoU and Dice scores were achieved on the validation set:

| Architecture | Mean IoU | Mean Dice |
|--------------|----------|-----------|
| U-Net       | 0.632167 | 0.765653 |
| LinkNet     | 0.614253 | 0.754901 |
| DeepLabV3+  | 0.581693 | 0.727155 |

U-Net performed best, achieving the highest IoU and Dice scores.

### Qualitative Results
Visual results show effective subject isolation with clean masks after morphological post-processing. Before-and-after images demonstrate background removal capabilities.

### Lessons Learned
- Data augmentation improved model robustness.
- Morphological operations (opening/closing) enhanced mask quality.
- U-Net provided reliable performance for subject segmentation.
- Challenges: Varying subject sizes; solutions: Multi-scale augmentations.
- Deployment: Flask integration required careful image handling.

## Project Structure

```
VisionExtract/
├── app.py                          # Flask web application
├── COCO.ipynb                      # Dataset exploration notebook
├── milestone2.ipynb                # Initial training notebook
├── milestone3.ipynb                # Advanced training and inference notebook
├── requirements.txt                # Python dependencies
├── README.md                       # This documentation
├── Model/
│   └── best_unet_model.pth         # Trained model weights
├── templates/
│   └── index.html                  # Web UI template
├── Screenshots/                    # Visual results and documentation
│   ├── milestone1/
│   └── milestone2/
└── data/                           # COCO dataset (download separately)
```

## Contributing

Contributions are welcome! Please fork the repo and submit a PR with your changes. Focus on improving model accuracy, adding new augmentations, or enhancing the UI.

## Acknowledgments

- COCO 2017 Dataset for training data.
- PyTorch and Segmentation Models library for implementation.
- Built as part of Infosys Springboard program.

Check the `Screenshots/` folder for weekly deliverables and visual results.
