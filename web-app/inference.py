"""
VisionExtract - Inference Module
Loads a trained model and runs subject isolation on new images.

Usage:
    # Single image
    python inference.py --image photo.jpg --checkpoint checkpoints/best_model.pth

    # Entire folder
    python inference.py --image_dir ./test_images --checkpoint checkpoints/best_model.pth

Output: for each input image, produces a new image where only the detected
        subject is visible and all background pixels are set to black.
"""

import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import VisionExtractUNet


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_inference_transform(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────
# Model Loader
# ─────────────────────────────────────────────

def load_model(checkpoint_path, device, use_attention=False):
    """Load trained VisionExtractUNet from a checkpoint."""
    model = VisionExtractUNet(pretrained=False, use_attention=use_attention)
    ckpt  = torch.load(checkpoint_path, map_location=device)

    # Support both raw state_dict and our training checkpoint format
    state_dict = ckpt.get('model_state', ckpt)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    cfg = ckpt.get('cfg', {})
    print(f"[Inference] Model loaded from: {checkpoint_path}")
    if cfg:
        print(f"            Trained for {cfg.get('epochs', '?')} epochs, "
              f"best IoU: {ckpt.get('best_iou', 'N/A'):.4f}")
    return model


# ─────────────────────────────────────────────
# Core Inference Function
# ─────────────────────────────────────────────

@torch.no_grad()
def predict_mask(model, image_rgb: np.ndarray, device,
                 img_size=256, threshold=0.5) -> np.ndarray:
    """
    Run the model on a single image and return a binary mask.

    Args:
        model     : Loaded VisionExtractUNet
        image_rgb : HxWx3 numpy array (uint8, RGB)
        device    : torch device
        img_size  : Model input size
        threshold : Probability threshold for foreground classification

    Returns:
        mask : HxW numpy array (uint8), 255=subject, 0=background
    """
    orig_h, orig_w = image_rgb.shape[:2]
    transform = get_inference_transform(img_size)

    tensor = transform(image=image_rgb)['image']          # (3, img_size, img_size)
    tensor = tensor.unsqueeze(0).to(device)               # (1, 3, H, W)

    prob = model(tensor)                                   # (1, 1, H, W) in [0,1]

    # Resize prediction back to original image dimensions
    prob_resized = F.interpolate(prob, size=(orig_h, orig_w),
                                 mode='bilinear', align_corners=False)
    mask = (prob_resized.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255
    return mask


def apply_mask(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply binary mask to image:  subject pixels stay, background → black.

    Args:
        image_rgb : HxWx3 numpy array (uint8)
        mask      : HxW numpy array (uint8), 255=subject

    Returns:
        result : HxWx3 numpy array with background zeroed out
    """
    binary = (mask > 127).astype(np.uint8)          # 0 or 1
    result = image_rgb * binary[:, :, np.newaxis]   # broadcast across channels
    return result


# ─────────────────────────────────────────────
# File I/O Helpers
# ─────────────────────────────────────────────

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def get_image_paths(image_dir):
    """Return all supported image paths in a directory."""
    paths = []
    for fname in sorted(os.listdir(image_dir)):
        if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTS:
            paths.append(os.path.join(image_dir, fname))
    return paths


def process_single_image(model, img_path, output_dir, device,
                          img_size=256, threshold=0.5, save_mask=False):
    """
    Run inference on one image and save outputs.

    Saves:
        <output_dir>/<name>_isolated.png  — subject on black background
        <output_dir>/<name>_mask.png      — binary mask (if save_mask=True)
    """
    image_pil = Image.open(img_path).convert('RGB')
    image_rgb = np.array(image_pil)

    mask   = predict_mask(model, image_rgb, device, img_size, threshold)
    result = apply_mask(image_rgb, mask)

    base   = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, f'{base}_isolated.png')
    Image.fromarray(result).save(out_path)

    if save_mask:
        mask_path = os.path.join(output_dir, f'{base}_mask.png')
        Image.fromarray(mask).save(mask_path)

    return out_path, mask


# ─────────────────────────────────────────────
# Batch Inference
# ─────────────────────────────────────────────

def run_inference(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Inference] Using device: {device}")

    model = load_model(cfg.checkpoint, device,
                       use_attention=cfg.attention)

    # Collect images to process
    if cfg.image:
        img_paths = [cfg.image]
    elif cfg.image_dir:
        img_paths = get_image_paths(cfg.image_dir)
        print(f"[Inference] Found {len(img_paths)} images in {cfg.image_dir}")
    else:
        raise ValueError("Provide --image or --image_dir")

    results = []
    for i, img_path in enumerate(img_paths, 1):
        print(f"  [{i}/{len(img_paths)}] Processing: {os.path.basename(img_path)}")
        try:
            out_path, _ = process_single_image(
                model, img_path,
                output_dir=cfg.output_dir,
                device=device,
                img_size=cfg.img_size,
                threshold=cfg.threshold,
                save_mask=cfg.save_mask,
            )
            results.append({'input': img_path, 'output': out_path, 'status': 'ok'})
            print(f"    → Saved: {out_path}")
        except Exception as e:
            print(f"    ✗ Error: {e}")
            results.append({'input': img_path, 'status': 'error', 'error': str(e)})

    ok    = sum(1 for r in results if r['status'] == 'ok')
    print(f"\n[Inference] Done. {ok}/{len(results)} images processed successfully.")
    print(f"[Inference] Outputs saved to: {cfg.output_dir}")
    return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='VisionExtract Inference')
    group  = parser.add_mutually_exclusive_group()
    group.add_argument('--image',       type=str, help='Path to a single input image')
    group.add_argument('--image_dir',   type=str, help='Path to folder of images')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save isolated images')
    parser.add_argument('--img_size',   type=int, default=256,
                        help='Model input size (should match training)')
    parser.add_argument('--threshold',  type=float, default=0.5,
                        help='Mask binarization threshold (0-1)')
    parser.add_argument('--attention',  action='store_true',
                        help='Use attention gates (must match training config)')
    parser.add_argument('--save_mask',  action='store_true',
                        help='Also save the raw binary mask')
    return parser.parse_args()


if __name__ == '__main__':
    cfg = parse_args()
    run_inference(cfg)
