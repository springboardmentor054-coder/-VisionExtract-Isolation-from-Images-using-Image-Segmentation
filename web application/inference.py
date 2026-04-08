import torch
import numpy as np
import cv2
from torchvision import models, transforms


# ─────────────────────────────────────────────
# Load Model (Pretrained - NO .pth needed)
# ─────────────────────────────────────────────

def load_model(checkpoint_path=None, device='cpu', use_attention=False):
    model = models.segmentation.deeplabv3_resnet101(weights="DEFAULT")
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────
# Preprocess
# ─────────────────────────────────────────────

def preprocess(image, img_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


# ─────────────────────────────────────────────
# Predict Mask
# ─────────────────────────────────────────────

def predict_mask(model, image_rgb, device, img_size=256, threshold=0.5):
    input_tensor = preprocess(image_rgb, img_size).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]

    # COCO class 15 = person
    mask = output.argmax(0).byte().cpu().numpy()
    mask = (mask == 15).astype(np.uint8) * 255

    # Resize to original
    mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))

    return mask


# ─────────────────────────────────────────────
# Apply Mask
# ─────────────────────────────────────────────

def apply_mask(image_rgb, mask):
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=2)

    result = image_rgb * mask
    return result.astype(np.uint8)
