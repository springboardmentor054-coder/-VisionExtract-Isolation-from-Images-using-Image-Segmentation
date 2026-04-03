# VisionExtract 🎯
**Automated Subject Isolation from Images using Deep Learning Segmentation**

> Upload any image → get back the main subject with a clean black background.  
> Powered by a **ResNet34-UNet** trained on **COCO 2017**.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Setup & Installation](#3-setup--installation)
4. [Dataset Preparation](#4-dataset-preparation)
5. [Training Guide](#5-training-guide)
6. [Running Inference](#6-running-inference)
7. [Web App](#7-web-app)
8. [Metrics & Results](#8-metrics--results)
9. [Project Structure](#9-project-structure)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Project Overview

VisionExtract detects and extracts the main subject from any image. Given an input photo, the model outputs a new image where:
- **Subject pixels** — kept exactly as in the original
- **Background pixels** — replaced with black (RGB 0, 0, 0)

This "cutout" functionality is used in photography automation, AR/VR, virtual conferencing, and media editing pipelines.

---

## 2. Architecture

```
Input Image (256×256×3)
        │
   ┌────▼────────────────────────────┐
   │     ResNet34 Encoder            │
   │   stem → layer1 → layer2        │
   │        → layer3 → layer4        │  ← ImageNet pretrained
   └─────────────────────────────────┘
   Skip connections: e0, e1, e2, e3
        │
   ┌────▼────────────────────────────┐
   │     UNet Decoder                │
   │  DecoderBlock × 4               │
   │  (Upsample + concat skip + Conv)│
   │  Optional: Attention Gates      │
   └─────────────────────────────────┘
        │
   ┌────▼────────────────────────────┐
   │  Final Head: Conv1×1 + Sigmoid  │
   └─────────────────────────────────┘
        │
   Output Mask (256×256×1), values in [0, 1]
        │
   Threshold (0.5) → Binary Mask → Apply to original image
```

**Why ResNet34-UNet?**
- ResNet34 encoder gives strong pretrained feature extraction with moderate compute
- UNet skip connections preserve fine spatial details crucial for clean cutouts
- Binary segmentation task is well-suited for this architecture
- ~21M parameters — good accuracy/speed balance

**Loss Function:** Combined Dice + BCE (50/50 weight)
- **BCE** — handles pixel-level boundary precision
- **Dice** — optimises mask overlap directly (aligns with IoU metric)

---

## 3. Setup & Installation

### Requirements
- Python 3.9+
- CUDA-capable GPU (recommended; CPU works but training is slow)
- 8GB+ RAM, 4GB+ VRAM (for batch_size=16)

### Step 1 — Clone / set up folder
```bash
mkdir visionextract && cd visionextract
# Place all .py files here
```

### Step 2 — Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install albumentations pycocotools fastapi uvicorn[standard] pillow numpy tqdm tensorboard pydantic python-multipart
```

> **Windows note:** `pycocotools` may need `pip install pycocotools-windows` instead.

### Verify GPU
```python
import torch
print(torch.cuda.is_available())   # Should print True
print(torch.cuda.get_device_name()) # e.g. NVIDIA RTX 3070
```

---

## 4. Dataset Preparation

### COCO 2017 Structure
Make sure your local COCO dataset looks like this:

```
/path/to/coco/
├── images/
│   ├── train2017/          ← ~118,000 images
│   └── val2017/            ←   ~5,000 images
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

### Update the path
In `train.py`, set `--coco_root` to your COCO path:
```bash
python train.py --coco_root /path/to/your/coco
```

### How masks are built
The COCO dataset has **instance-level** annotations (each object separately). VisionExtract merges all instance masks into a single **binary** mask:
- Any pixel belonging to ANY annotated object → **foreground (1)**
- All other pixels → **background (0)**

This is done automatically in `dataset.py` via `coco.annToMask()`.

### Quick sanity check
```bash
python dataset.py
# Outputs sample_check.png showing one image + its binary mask
```

---

## 5. Training Guide

### Quick start (validation set, fast experiment)
```bash
python train.py \
  --coco_root /path/to/your/coco \
  --max_train 5000 \
  --max_val   1000 \
  --epochs 10 \
  --batch_size 16
```

### Full training (recommended)
```bash
python train.py \
  --coco_root /path/to/your/coco \
  --img_size 256 \
  --batch_size 16 \
  --epochs 30 \
  --lr 1e-4 \
  --attention \
  --checkpoint_dir checkpoints
```

### Key training arguments

| Argument | Default | Description |
|---|---|---|
| `--coco_root` | *(required)* | Path to COCO 2017 root |
| `--img_size` | 256 | Input resolution (try 320 for better quality) |
| `--batch_size` | 16 | Reduce to 8 if OOM |
| `--epochs` | 30 | Training epochs |
| `--lr` | 1e-4 | Initial learning rate (AdamW) |
| `--attention` | False | Enable attention gates (improves ~1-2% IoU) |
| `--patience` | 7 | Early stopping patience |
| `--max_train` | None | Limit training samples (for quick tests) |

### Monitor with TensorBoard
```bash
tensorboard --logdir runs/
# Open http://localhost:6006
```

### What good training looks like
- **Epoch 1-5:** IoU rises quickly from ~0.3 → ~0.6
- **Epoch 5-15:** Steady improvement, IoU ~0.6 → ~0.72
- **Epoch 15-30:** Fine-tuning, IoU plateaus ~0.73-0.78
- Early stopping triggers if validation IoU stops improving for 7 epochs

### Resuming training
The latest checkpoint is always saved as `checkpoints/latest.pth`. To resume:
```python
# Add to train.py (coming improvement) or load manually:
ckpt = torch.load('checkpoints/latest.pth')
model.load_state_dict(ckpt['model_state'])
optimizer.load_state_dict(ckpt['optim_state'])
scheduler.load_state_dict(ckpt['scheduler'])
```

---

## 6. Running Inference

### Single image
```bash
python inference.py \
  --image my_photo.jpg \
  --checkpoint checkpoints/best_model.pth \
  --output_dir outputs/
```

### Entire folder
```bash
python inference.py \
  --image_dir ./test_photos/ \
  --checkpoint checkpoints/best_model.pth \
  --output_dir outputs/ \
  --save_mask   # also save binary masks
```

### Adjust threshold
The `--threshold` parameter (default 0.5) controls how aggressively the background is cut:
- **Lower (0.3)** → keeps more pixels (less aggressive, safer for fuzzy edges)
- **Higher (0.7)** → removes more pixels (cleaner background, may clip subject edges)

```bash
python inference.py --image photo.jpg --checkpoint ... --threshold 0.35
```

### Python API
```python
from PIL import Image
import numpy as np
import torch
from inference import load_model, predict_mask, apply_mask

device = torch.device('cuda')
model  = load_model('checkpoints/best_model.pth', device)

image_rgb = np.array(Image.open('photo.jpg').convert('RGB'))
mask      = predict_mask(model, image_rgb, device, img_size=256, threshold=0.5)
result    = apply_mask(image_rgb, mask)

Image.fromarray(result).save('isolated.png')
```

---

## 7. Web App

### Start the backend API
```bash
# Set your checkpoint path
export CHECKPOINT_PATH=checkpoints/best_model.pth

uvicorn backend:app --host 0.0.0.0 --port 8000
```

API docs available at: http://localhost:8000/docs

### Open the frontend
Simply open `frontend/index.html` in your browser. No build step needed.

Or serve it with a simple HTTP server:
```bash
cd frontend
python -m http.server 3000
# Open http://localhost:3000
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check API and model status |
| `POST` | `/predict` | Upload image → receive PNG blob |
| `POST` | `/predict/base64` | Upload image → receive base64 JSON |

### Example API call (curl)
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@my_photo.jpg" \
  -F "threshold=0.5" \
  --output isolated.png
```

---

## 8. Metrics & Results

### Primary Metric: Intersection over Union (IoU)

```
        |Predicted ∩ Ground Truth|
IoU = ─────────────────────────────────
        |Predicted ∪ Ground Truth|
```

- **Range:** 0 (no overlap) → 1 (perfect overlap)
- **Target:** IoU ≥ 0.70 on COCO val set

### Additional Metrics

**Dice Coefficient:**
```
        2 × |Predicted ∩ Ground Truth|
Dice = ─────────────────────────────────────────
        |Predicted| + |Ground Truth|
```
More sensitive to small objects than IoU. Target: ≥ 0.78

**Pixel Accuracy:**
```
              Correct pixels
Acc = ─────────────────────────────
              Total pixels
```
Can be misleadingly high with class imbalance (large backgrounds). Use alongside IoU.

### Expected Benchmark Performance (COCO val2017)

| Model | IoU | Dice | Pixel Acc |
|---|---|---|---|
| ResNet34-UNet (no attention) | ~0.70 | ~0.78 | ~0.92 |
| ResNet34-UNet + Attention | ~0.72 | ~0.80 | ~0.93 |

> *Results depend on training duration, data volume, and hardware. With full COCO train set + 30 epochs, expect IoU in 0.68–0.75 range.*

### Interpreting Results

| IoU | Interpretation |
|---|---|
| < 0.50 | Poor — model is not learning well |
| 0.50–0.65 | Acceptable — basic separation working |
| 0.65–0.75 | Good — clean cutouts in most cases |
| > 0.75 | Excellent — production-ready |

### Loss Curve Interpretation
- Training loss should decrease smoothly
- Validation loss should follow closely (large gap = overfitting)
- If validation loss increases while training loss decreases → overfitting → add augmentation or reduce epochs

---

## 9. Project Structure

```
visionextract/
├── dataset.py       # COCO loading, binary mask generation, augmentation
├── model.py         # ResNet34-UNet architecture + loss functions
├── train.py         # Training loop, metrics, checkpointing, early stopping
├── inference.py     # Single/batch image inference
├── backend.py       # FastAPI REST API server
├── frontend/
│   └── index.html   # Web UI (drag-drop, threshold control, download)
├── checkpoints/     # Saved model weights (auto-created during training)
│   ├── best_model.pth
│   └── latest.pth
├── runs/            # TensorBoard logs (auto-created during training)
└── outputs/         # Inference results (auto-created)
```

---

## 10. Troubleshooting

**CUDA out of memory**
→ Reduce `--batch_size` to 8 or 4. Or reduce `--img_size` to 224.

**pycocotools install fails on Windows**
→ `pip install pycocotools-windows`

**`ModuleNotFoundError: No module named 'dataset'`**
→ Run all scripts from the `visionextract/` directory, not from subdirectories.

**API returns 503 "Model not loaded"**
→ Check that `CHECKPOINT_PATH` env var points to a valid `.pth` file. Run training first.

**Frontend can't reach API (CORS error)**
→ The backend has CORS enabled for all origins. Make sure the API is running on port 8000.

**Mask cuts off subject edges**
→ Lower `--threshold` to 0.35–0.4. Or increase training data augmentation variety.

**Training IoU stuck at ~0.3**
→ Check that masks are loading correctly (`python dataset.py` to visualize). Ensure correct COCO path.
