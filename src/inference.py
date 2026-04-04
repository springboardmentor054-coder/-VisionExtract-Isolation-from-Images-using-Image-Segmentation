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
    def __init__(self, model_path=None, device=None, image_size=256):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = 256 # Fixed to 256 for stable results
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
        self.valid_formats = (".jpg", ".png", ".jpeg")

    def full_pipeline(self, image_path, output_path=None, save=True, display=False, custom_size=None):
        """Standard segmentation pipeline: Preprocess -> Predict -> Crop -> Upscale."""
        if not image_path.lower().endswith(self.valid_formats):
            raise ValueError(f"Invalid image format: {image_path}. Supported: {self.valid_formats}")

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image.shape[:2]
        
        # Determine image size for inference
        target_size = custom_size if custom_size else self.image_size
        transforms = get_val_transforms(image_size=target_size)
        
        # 1. Preprocess
        augmented = transforms(image=image)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device)

        # 2. Prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            # Use raw probabilities for smooth alpha-matting
            prediction = torch.sigmoid(output).squeeze().cpu().numpy()

        # 3. Handle Padding Adjustment 
        # Standard centering logic to undo PadIfNeeded (from dataset.py)
        scale = target_size / max(h_orig, w_orig)
        new_h, new_w = int(h_orig * scale), int(w_orig * scale)
        pad_top = (target_size - new_h) // 2
        pad_left = (target_size - new_w) // 2
        
        # Extract correctly aligned valid region
        valid_mask = prediction[pad_top:pad_top+new_h, pad_left:pad_left+new_w]

        # 4. Final Upscale & Matting
        # Resize mask back to original resolution for smooth, high-fidelity isolation
        final_mask = cv2.resize(valid_mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        
        # Apply mask to isolate subject
        isolated = (image * final_mask[:, :, None]).astype(np.uint8)

        # Save Output
        if save:
            if not output_path:
                timestamp = str(int(time.time()))
                output_folder = "outputs"
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, f"isolated_{timestamp}.png")
            else:
                os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            
            output_image = Image.fromarray(isolated)
            output_image.save(output_path)

        if display:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1); plt.imshow(image); plt.title("Original")
            plt.subplot(1, 2, 2); plt.imshow(isolated); plt.title("Isolated (Standard)")
            plt.show()

        return isolated, final_mask

    def batch_inference(self, folder_path, output_dir="outputs"):
        """Process all images in a folder using the standard pipeline."""
        if not os.path.exists(folder_path):
            logger.error(f"Folder {folder_path} does not exist.")
            return

        images = [f for f in os.listdir(folder_path) if f.lower().endswith(self.valid_formats)]
        logger.info(f"Found {len(images)} images. Processing batch...")

        os.makedirs(output_dir, exist_ok=True)
        start_time = time.time()
        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            out_path = os.path.join(output_dir, f"isolated_{img_name.split('.')[0]}.png")
            try:
                self.full_pipeline(img_path, output_path=out_path, save=True, display=False)
            except Exception as e:
                logger.error(f"Error processing {img_name}: {e}")

        logger.info(f"Batch completed in: {time.time() - start_time:.2f}s.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VisionExtract Subject Isolation CLI")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--dir", type=str, help="Path to a directory for batch processing")
    parser.add_argument("--output", type=str, help="Specify output path for single image")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for batch")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--size", type=int, default=256, help="Inference resolution")
    parser.add_argument("--display", action="store_true", help="Display results using Matplotlib")
    
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
                    # Sort to get latest epoch
                    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'epoch' in x else 0)
                    model_path = os.path.join(checkpoint_dir, checkpoints[-1])
            
    pipeline = VisionExtractPipeline(model_path=model_path, image_size=args.size)
    
    if args.image:
        pipeline.full_pipeline(args.image, output_path=args.output, save=True, display=args.display)
    elif args.dir:
        pipeline.batch_inference(args.dir, output_dir=args.output_dir)
    else:
        print("Usage: python src/inference.py --image <path> [--output <path>] OR --dir <path>")
