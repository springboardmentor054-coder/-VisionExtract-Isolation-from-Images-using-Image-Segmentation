import sys
import os
import time

# Redirect PyTorch and XDG caches to E: drive to prevent C: drive filling up
os.environ['TORCH_HOME'] = r'E:\torch_cache'
os.environ['XDG_CACHE_HOME'] = r'E:\torch_cache'

# Ensure src directory is in path for local imports
sys.path.append(os.path.dirname(__file__))

import torch
import numpy as np
import cv2
import logging
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import UNet
from dataset import get_val_transforms

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VisionExtractPipeline:
    def __init__(self, model_path=None, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet().to(self.device)
        
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load weight: {e}. Using random weights.")
        else:
            logger.warning("No model path provided or file doesn't exist. Model is uninitialized.")
            
        self.model.eval()
        self.transforms = get_val_transforms()
        self.valid_formats = (".jpg", ".png", ".jpeg")

    def clean_mask(self, mask):
        """Refine the predicted mask using morphological operations."""
        mask_uint8 = (mask * 255).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        return cleaned.astype(float) / 255.0

    def full_pipeline(self, image_path, output_path=None, save=True, display=False):
        """End-to-end pipeline: load, preprocess, predict, clean, isolate, save."""
        if not image_path.lower().endswith(self.valid_formats):
            raise ValueError(f"Invalid image format: {image_path}. Supported: {self.valid_formats}")

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        augmented = self.transforms(image=image)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device)

        # Prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = torch.sigmoid(output)
            mask = (prediction > 0.5).float().squeeze().cpu().numpy()

        # Clean Mask
        cleaned_mask = self.clean_mask(mask)

        # Isolate Subject
        resized_original = cv2.resize(image, (256, 256))
        isolated = (resized_original * cleaned_mask[:, :, None]).astype(np.uint8)

        # Save Output
        if save:
            if not output_path:
                timestamp = str(int(time.time()))
                output_folder = "outputs"
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, f"output_{timestamp}.png")
            else:
                os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            
            output_image = Image.fromarray(isolated)
            output_image.save(output_path)
            logger.info(f"Output saved: {output_path}")

        if display:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(resized_original)
            plt.title("Original (Resized)")
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(isolated)
            plt.title("Isolated Subject")
            plt.axis("off")
            plt.show()

        return isolated

    def batch_inference(self, folder_path, output_dir="outputs"):
        """Process all images in a folder."""
        if not os.path.exists(folder_path):
            logger.error(f"Folder {folder_path} does not exist.")
            return

        images = [f for f in os.listdir(folder_path) if f.lower().endswith(self.valid_formats)]
        logger.info(f"Found {len(images)} images in {folder_path}. Starting batch inference...")

        os.makedirs(output_dir, exist_ok=True)
        start_time = time.time()
        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            out_path = os.path.join(output_dir, f"isolated_{img_name.split('.')[0]}.png")
            try:
                self.full_pipeline(img_path, output_path=out_path, save=True, display=False)
            except Exception as e:
                logger.error(f"Error processing {img_name}: {e}")

        logger.info(f"Batch inference completed in: {time.time() - start_time:.2f}s.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VisionExtract Subject Isolation CLI")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--dir", type=str, help="Path to a directory for batch processing")
    parser.add_argument("--output", type=str, help="Specify output path for single image")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for batch")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--display", action="store_true", help="Display result")
    
    args = parser.parse_args()

    # Model weight detection
    model_path = args.checkpoint
    if not model_path:
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
            if checkpoints:
                if "best_model.pth" in checkpoints:
                    model_path = os.path.join(checkpoint_dir, "best_model.pth")
                else:
                    # Sort by epoch number to get the latest
                    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'epoch' in x else 0)
                    model_path = os.path.join(checkpoint_dir, checkpoints[-1])
            
    pipeline = VisionExtractPipeline(model_path=model_path)
    
    if args.image:
        pipeline.full_pipeline(args.image, output_path=args.output, save=True, display=args.display)
    elif args.dir:
        pipeline.batch_inference(args.dir, output_dir=args.output_dir)
    else:
        print("Usage: python src/inference.py --image <path> [--output <path>] OR --dir <path> [--output_dir <path>]")