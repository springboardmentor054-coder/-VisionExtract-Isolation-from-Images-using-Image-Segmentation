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

The model employs a deep learning-based semantic segmentation network (e.g., U-Net or a variant like DeepLab) to generate pixel-wise binary masks. During inference, the mask is applied to retain original pixels for the subject and set background pixels to black.

### High-Level Pipeline Diagram

```
+----------------+     +----------------+     +----------------+     +-----------------+
|   Web App      |     |   HTTP Request |     |   Image Encoder|     | Segmentation    |
| (User Upload)  | --> | (Backend API)  | --> | (CNN Backbone) | --> | Model (U-Net)   |
|                |     |                |     |                |     |                 |
+----------------+     +----------------+     +----------------+     +-----------------+
                                                                  |
                                                                  v
+----------------+     +----------------+     +----------------+     +-----------------+
|   Flask/FastAPI| <-- |   Result Mask  | <-- |   Decoder      | <-- |   Output Image  |
| (Backend Server|     | (Binary: Subject|     | (Upsampling)   |     | (Subject Only)  |
| on Cloud)      |     | vs Background) |     |                |     |                 |
+----------------+     +----------------+     +----------------+     +-----------------+
```

- **Input**: RGB image (e.g., 512x512 resolution).
- **Output**: Isolated subject image with black background.
- **Backbone**: ResNet or MobileNet for feature extraction.
- **Deployment**: Hosted on a cloud server (e.g., AWS/GCP) via Flask/FastAPI.

## Modules to Implement

### 1. Data Preprocessing and Feature Engineering
- Handle raw image and mask data preparation for consistency.
- Apply data augmentation (cropping, flipping, color perturbation), normalization.
- Convert masks to binary subject-background format.
- Ensure alignment between images and masks for accurate training.

### 2. Building the Segmentation Model
- Predict pixel-wise binary masks separating subject from background.
- During inference: Retain input pixels for subject; replace background with black.

### 3. Evaluation and Fine-Tuning
- Quantitative metrics: IoU, Dice Coefficient, Precision, Recall, Pixel-wise Accuracy.
- Qualitative review: Inspect sample output images for visual fidelity.
- Fine-tune with hyperparameters or additional augmentations.

### 4. Documentation and Presentation Preparation
- Document dataset handling, pipeline, experiments, and results.
- Include before/after visuals and explain challenges/solutions.

## Implementation Milestones

The project is structured over 8 weeks with weekly deliverables and screenshots of outputs.

### Milestone 1: Project Initialization and Data Setup
- **Week 1: Project Initialization and Dataset Acquisition**
  - Define objectives and acquire/inspect COCO 2017 dataset.
  - Explore structure; view example images and masks.
- **Week 2: Data Preprocessing and Validation**
  - Build preprocessing pipeline (resize, normalization, augmentation).
  - Ensure image-mask alignment; convert multi-class to binary masks.

### Milestone 2: Initial Model Training
- **Week 3: Initial Model Training**
  - Implement segmentation network (e.g., U-Net).
  - Train on processed data; monitor metrics.
  - Visualize early predictions.
- **Week 4: Predictions and Fine-Tuning**
  - Generate validation predictions; compare with ground-truth.
  - Adjust hyperparameters and augmentations.

### Milestone 3: Improvements and Inference
- **Week 5: Improve Data and Experiment with Architectures**
  - Refine preprocessing; explore post-processing for cleaner masks.
  - Test variant architectures; document comparisons.
- **Week 6: Inference**
  - Deploy model for new images.
  - Automate pipeline; test on unseen data for robustness.

### Milestone 4: Full Pipeline and Demo
- **Week 7: Full Pipeline and User Interface**
  - Build web app for image upload and result download.
  - Integrate preprocessing, inference, and output generation.
- **Week 8: Documentation, Presentation, and Demo**
  - Compile docs with results and lessons learned.
  - Prepare presentation with visuals; conduct live demo.

## Evaluation Criteria

1. **Completion of Milestones**: Successful data handling, model development, inference, and UI integration within timelines.
2. **Accuracy of Subject Isolation**: Quantitative (IoU, Dice) and qualitative assessment of separation quality.
3. **Clarity and Depth of Documentation and Presentation**: Completeness of docs; effective communication of process, results, and before/after examples.

## Results and Evaluation

### Quantitative Results
After training and evaluating three architectures (U-Net, LinkNet, DeepLabV3+), the following mean IoU and Dice scores were achieved on the validation set:

| Architecture | Mean IoU | Mean Dice |
|--------------|----------|-----------|
| U-Net       | 0.85     | 0.92     |
| LinkNet     | 0.83     | 0.91     |
| DeepLabV3+  | 0.87     | 0.93     |

DeepLabV3+ performed best, achieving the highest IoU and Dice scores.

### Qualitative Results
Visual comparisons showed that all models effectively isolated subjects, with DeepLabV3+ providing the cleanest masks after morphological post-processing. Before-and-after images demonstrate the model's ability to remove backgrounds accurately.

### Lessons Learned
- Data augmentation significantly improved model generalization.
- Morphological post-processing (opening and closing) enhanced mask quality by removing noise and filling holes.
- U-Net provided a good balance of simplicity and performance; DeepLabV3+ excelled in capturing fine details.
- Challenges included handling varying subject sizes and complex backgrounds; solutions involved refined augmentations and multi-scale training.
- For deployment, integrating the model into a Flask app required careful handling of image preprocessing and output formatting.

## Setup and Installation

1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run the web app: `python app.py`

## Usage

1. Ensure the trained model `best_unet_model.pth` is in the project root (run the training cells in milestone3.ipynb if not present).
2. Run `python app.py` to start the server.
3. Open `http://localhost:5000/` in a browser.
4. Upload an image to get the isolated subject image.

## Presentation and Demo

For the presentation, include:
- Project overview and objectives.
- Dataset and preprocessing details.
- Model architectures compared (U-Net, LinkNet, DeepLabV3+).
- Quantitative and qualitative results with before/after visuals.
- Challenges and solutions.
- Live demo of the web app.

Screenshots and results are available in the `Screenshots/` folder.

## Project Structure

### Subject Segmentation
- **Metric**: Intersection over Union (IoU)
  - **Description**: Measures overlap between predicted subject region and ground-truth mask divided by their union. High IoU indicates better segmentation accuracy.

### Additional Metrics
- **Metric**: Dice Coefficient, Pixel-wise Accuracy
  - **Description**: Measures alignment and quality of predicted masks, especially useful for binary separation tasks.

### Example Quantitative Goals
1. **Subject Isolation Performance**
   - **Goal**: Achieve high IoU and Dice scores on validation images, indicating the model reliably separates the subject from the background.


## Project Structure
```
visionextract/
├── data/                  # Dataset and processed data
├── src/                   # Source code
│   ├── preprocess.py      # Data pipeline
│   ├── model.py           # Segmentation model
│   ├── train.py           # Training script
│   └── inference.py       # Inference script
├── app.py                 # Web app
├── requirements.txt       # Dependencies
├── screenshots/           # Documentation and screenshots
└── README.md              # This file
```

## Contributing

Contributions are welcome! Please fork the repo and submit a PR with your changes. Focus on improving model accuracy, adding new augmentations, or enhancing the UI.

## Acknowledgments

- Inspired by COCO dataset and segmentation best practices.
- Built with PyTorch and Segmentation Models library.

 Check the `screenshot/` folder for weekly screenshots and full reports.
