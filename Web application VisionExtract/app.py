"""
VisionExtract - FastAPI Backend
REST API that accepts image uploads and returns subject-isolated results.

Endpoints:
    POST /predict          - Upload image, get isolated image back
    POST /predict/base64   - Upload image, get base64 response
    GET  /health           - Health check

Run:
    uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import io
import base64
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from PIL import Image
import torch

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our inference utilities
import sys
sys.path.insert(0, os.path.dirname(__file__))
from inference import load_model, predict_mask, apply_mask

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', 'checkpoints/best_model.pth')
IMG_SIZE        = int(os.environ.get('IMG_SIZE', '256'))
THRESHOLD       = float(os.environ.get('THRESHOLD', '0.5'))
USE_ATTENTION   = os.environ.get('USE_ATTENTION', 'false').lower() == 'true'
MAX_UPLOAD_MB   = 10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('visionextract')

# ─────────────────────────────────────────────
# Global Model State
# ─────────────────────────────────────────────

model_state = {'model': None, 'device': None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Starting VisionExtract API | device={device}")

    try:
       logger.info("Loading pretrained segmentation model...")
       model_state['model']  = load_model(None, device, USE_ATTENTION)
       model_state['device'] = device
       logger.info("Model loaded successfully (pretrained)")
    except Exception as e:
         logger.error(f"Failed to load model: {e}")

    yield  # App runs here

    model_state['model'] = None
    logger.info("VisionExtract API shutdown.")


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title='VisionExtract API',
    description='Subject isolation from images using deep learning segmentation.',
    version='1.0.0',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],     # Tighten this in production
    allow_methods=['*'],
    allow_headers=['*'],
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def validate_upload(file: UploadFile):
    allowed = {'image/jpeg', 'image/png', 'image/webp', 'image/bmp'}
    if file.content_type not in allowed:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}. "
                                 f"Allowed: {allowed}")


async def read_image(file: UploadFile) -> np.ndarray:
    """Read uploaded file bytes → RGB numpy array."""
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max {MAX_UPLOAD_MB}MB.")
    try:
        pil = Image.open(io.BytesIO(data)).convert('RGB')
        return np.array(pil)
    except Exception:
        raise HTTPException(400, "Could not decode image. Is it a valid image file?")


def get_model():
    if model_state['model'] is None:
        raise HTTPException(503, "Model not loaded. Check server logs.")
    return model_state['model'], model_state['device']


def image_to_bytes(image_rgb: np.ndarray, fmt='PNG') -> bytes:
    buf = io.BytesIO()
    Image.fromarray(image_rgb).save(buf, format=fmt)
    return buf.getvalue()


# ─────────────────────────────────────────────
# Response Schemas
# ─────────────────────────────────────────────

class PredictBase64Response(BaseModel):
    isolated_image_b64: str
    mask_b64: Optional[str] = None
    orig_width: int
    orig_height: int
    message: str = 'success'


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get('/health')
def health():
    """Check API and model status."""
    model_ok = model_state['model'] is not None
    return {
        'status'      : 'ok' if model_ok else 'model_not_loaded',
        'model_loaded': model_ok,
        'device'      : str(model_state.get('device', 'N/A')),
        'checkpoint'  : CHECKPOINT_PATH,
    }


@app.post('/predict',
          response_class=Response,
          responses={200: {'content': {'image/png': {}},
                           'description': 'Isolated subject image (PNG)'}})
async def predict(
    file      : UploadFile = File(..., description='Input image (JPEG/PNG/WebP)'),
    threshold : float      = Form(0.5, ge=0.0, le=1.0,
                                  description='Mask threshold (default 0.5)')
):
    """
    Upload an image → receive the subject-isolated PNG back.
    Background pixels are set to black; subject pixels are unchanged.
    """
    validate_upload(file)
    model, device = get_model()

    image_rgb = await read_image(file)
    logger.info(f"Predicting on image {image_rgb.shape}, threshold={threshold}")

    mask      = predict_mask(model, image_rgb, device, IMG_SIZE, threshold)
    result    = apply_mask(image_rgb, mask)
    png_bytes = image_to_bytes(result, fmt='PNG')

    return Response(content=png_bytes, media_type='image/png')


@app.post('/predict/base64', response_model=PredictBase64Response)
async def predict_base64(
    file        : UploadFile = File(...),
    threshold   : float      = Form(0.5, ge=0.0, le=1.0),
    return_mask : bool       = Form(False, description='Also return the binary mask')
):
    """
    Upload an image → receive base64-encoded isolated image (and optionally the mask).
    Useful for direct browser/JavaScript integration without blob handling.
    """
    validate_upload(file)
    model, device = get_model()

    image_rgb = await read_image(file)
    h, w      = image_rgb.shape[:2]

    mask      = predict_mask(model, image_rgb, device, IMG_SIZE, threshold)
    result    = apply_mask(image_rgb, mask)

    isolated_b64 = base64.b64encode(image_to_bytes(result)).decode()
    mask_b64     = base64.b64encode(image_to_bytes(mask)).decode() if return_mask else None

    return PredictBase64Response(
        isolated_image_b64 = isolated_b64,
        mask_b64           = mask_b64,
        orig_width         = w,
        orig_height        = h,
    )

from utils import get_binary_mask, refine_mask


def isolate_subject(model, image, device):
    model.eval()

    with torch.no_grad():
        image = image.to(device)
        pred = model(image)

        mask = get_binary_mask(pred)

        # Optional: clean mask
        mask = refine_mask(mask)

        # Apply mask
        output = image * mask

    return output, mask
# ─────────────────────────────────────────────
# Run directly
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=True)
