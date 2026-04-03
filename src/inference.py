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
        self.transforms = get_val_transforms(image_size=self.image_size)
        self.valid_formats = (".jpg", ".png", ".jpeg")

    def _guided_filter(self, guide, src, radius, eps):
        """Standard Guided Filter implementation for edge refinement."""
        guide = guide.astype(np.float32) / 255.0
        src = src.astype(np.float32)
        
        if len(guide.shape) == 3:
            # Handle multi-channel guide (RGB)
            guide_gray = cv2.cvtColor((guide * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            guide_gray = guide

        mean_i = cv2.boxFilter(guide_gray, -1, (radius, radius))
        mean_p = cv2.boxFilter(src, -1, (radius, radius))
        mean_ip = cv2.boxFilter(guide_gray * src, -1, (radius, radius))
        cov_ip = mean_ip - mean_i * mean_p

        mean_ii = cv2.boxFilter(guide_gray * guide_gray, -1, (radius, radius))
        var_i = mean_ii - mean_i * mean_i

        a = cov_ip / (var_i + eps)
        b = mean_p - a * mean_i

        mean_a = cv2.boxFilter(a, -1, (radius, radius))
        mean_b = cv2.boxFilter(b, -1, (radius, radius))

        return mean_a * guide_gray + mean_b

    def clean_mask(self, mask, image_guide=None, refinement_intensity=0.5):
        """Refine the predicted mask using Morphological Erosion and Guided Filter."""
        if image_guide is not None:
            # Step 1: Halo Reduction (Morphological Erosion)
            # Contract the mask slightly before refinement to prevent background "bleed"
            kernel_size = int(3 + 2 * refinement_intensity)
            if kernel_size % 2 == 0: kernel_size += 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)

            # Step 2: Edge-Aware Smoothing (Guided Filter)
            # eps: regularization. Smaller eps snaps harder to guide edges. 1e-6 is ideal for hair.
            eps = 1e-6 + (1.0 - refinement_intensity) * 0.01
            radius = int(2 + 6 * refinement_intensity) # More local radius for sharper edges
            
            refined = self._guided_filter(image_guide, mask, radius, eps)
            refined = np.clip(refined, 0, 1)
            
            # Step 3: Post-refinement: Transparency & Background Suppression
            if refinement_intensity > 0.5:
                # Optimized gamma contrast for cleaner isolation
                refined = np.power(refined, 1.4) 
                refined[refined < 0.02] = 0 # Push very low probabilities to zero
                
            return refined
        else:
            return cv2.GaussianBlur(mask, (3, 3), 0)

    def full_pipeline(self, image_path, output_path=None, save=True, display=False, refinement=True, refinement_intensity=0.8, custom_size=None):
        """End-to-end pipeline with aspect ratio awareness and mask refinement."""
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
        
        # Preprocess
        augmented = transforms(image=image)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device)

        # Prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            # Use soft mask (probabilities) instead of binary for better upscaling
            prediction = torch.sigmoid(output).squeeze().cpu().numpy()

        # Handle Padding Adjustment: 
        # Albumentations PadIfNeeded pads to center by default or from edges.
        # Resize/Padding calculation to find the valid mask region
        # Handle Padding Adjustment
        scale = 256 / max(h_orig, w_orig)
        new_h, new_w = int(h_orig * scale), int(w_orig * scale)
        pad_top = (256 - new_h) // 2
        pad_left = (256 - new_w) // 2
        
        # Crop the valid region out of the square prediction
        valid_mask = prediction[pad_top:pad_top+new_h, pad_left:pad_left+new_w]

        # Upscale the mask back to original resolution BEFORE binarization for smooth edges
        upscaled_mask = cv2.resize(valid_mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        
        if refinement:
            upscaled_mask = self.clean_mask(upscaled_mask, image_guide=image, refinement_intensity=refinement_intensity)

        # Apply threshold to the refined soft mask
        binary_mask = (upscaled_mask > 0.5).astype(float)
        
        final_mask = upscaled_mask if refinement else binary_mask
        
        # Isolate Subject
        isolated = (image * final_mask[:, :, None]).astype(np.uint8)

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
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title(f"Original ({w_orig}x{h_orig})")
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(isolated)
            plt.title(f"Isolated Subject ({w_orig}x{h_orig})")
            plt.axis("off")
            plt.tight_layout()
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
            
    pipeline = VisionExtractPipeline(model_path=model_path, image_size=args.size)
    
    if args.image:
        pipeline.full_pipeline(args.image, output_path=args.output, save=True, display=args.display)
    elif args.dir:
        pipeline.batch_inference(args.dir, output_dir=args.output_dir)
    else:
        print("Usage: python src/inference.py --image <path> [--output <path>] OR --dir <path> [--output_dir <path>]")
