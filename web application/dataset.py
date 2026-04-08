"""
VisionExtract - Dataset Module
Handles COCO 2017 dataset loading, binary mask generation, and augmentation.

COCO 2017 expected folder structure:
    /path/to/coco/
        images/
            train2017/
            val2017/
        annotations/
            instances_train2017.json
            instances_val2017.json
"""

import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────
# Augmentation Pipelines
# ─────────────────────────────────────────────

def get_train_transforms(img_size=256):
    """Augmentations for training: flips, color jitter, elastic distort."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomBrightnessContrast(p=0.4),
        A.HueSaturationValue(p=0.3),
        A.ElasticTransform(p=0.2),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=256):
    """Minimal transforms for validation: just resize + normalize."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────
# COCO Subject Segmentation Dataset
# ─────────────────────────────────────────────

class COCOSegmentationDataset(Dataset):
    """
    Loads COCO images and merges ALL instance masks into a single
    binary subject mask (foreground=1, background=0).

    Args:
        root_dir   : Path to COCO root (contains images/ and annotations/)
        split      : 'train' or 'val'
        img_size   : Resize dimension (square)
        transform  : Albumentations transform pipeline
        max_samples: Limit dataset size (useful for quick experiments)
    """

    def __init__(self, root_dir, split='val', img_size=256,
                 transform=None, max_samples=None):
        assert split in ('train', 'val'), "split must be 'train' or 'val'"

        self.img_dir = os.path.join(root_dir, 'images', f'{split}2017')
        ann_file = os.path.join(root_dir, 'annotations',
                                f'instances_{split}2017.json')

        print(f"[Dataset] Loading COCO {split} annotations from:\n  {ann_file}")
        self.coco = COCO(ann_file)

        # Only keep images that have at least one annotation
        all_img_ids = self.coco.getImgIds()
        self.img_ids = [
            img_id for img_id in all_img_ids
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]

        if max_samples:
            self.img_ids = self.img_ids[:max_samples]

        self.transform = transform or get_val_transforms(img_size)
        self.img_size = img_size
        print(f"[Dataset] {split} set: {len(self.img_ids)} images with annotations.")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        # Load image as RGB numpy array
        image = np.array(Image.open(img_path).convert('RGB'))

        # Build binary mask: merge all instance masks → 1 if any subject pixel
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        h, w = image.shape[:2]
        binary_mask = np.zeros((h, w), dtype=np.uint8)

        for ann in anns:
            m = self.coco.annToMask(ann)  # shape: (H, W) with 0/1
            binary_mask = np.maximum(binary_mask, m)

        # Apply transforms (handles resize + augmentation on both image and mask)
        augmented = self.transform(image=image, mask=binary_mask)
        image_tensor = augmented['image']             # (3, H, W) float32
        mask_tensor = augmented['mask'].float()       # (H, W) float32, 0.0 or 1.0

        return image_tensor, mask_tensor.unsqueeze(0)  # mask → (1, H, W)


# ─────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────

def get_dataloaders(root_dir, img_size=256, batch_size=16,
                    num_workers=4, max_train=None, max_val=None):

    train_ds = COCOSegmentationDataset(
        root_dir, split='train', img_size=img_size,
        transform=get_train_transforms(img_size),
        max_samples=max_train
    )

    # Use a slice of train as validation (no val2017 images needed)
    val_ds = COCOSegmentationDataset(
        root_dir, split='train', img_size=img_size,
        transform=get_val_transforms(img_size),
        max_samples=max_val or 1000   # use 1000 train images as val
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    COCO_ROOT = 'E:\\VisionExtract\\data'
    split = 'train'   # ← change this
    ds = COCOSegmentationDataset(COCO_ROOT, split='train', max_samples=5)
    img, mask = ds[0]

    print(f"Image shape : {img.shape}")   # (3, 256, 256)
    print(f"Mask shape  : {mask.shape}")  # (1, 256, 256)
    print(f"Mask unique : {mask.unique()}")  # tensor([0., 1.])

    # Visualise
    fig, axes = plt.subplots(1, 2)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    axes[0].imshow((img * std + mean).permute(1,2,0).clamp(0,1))
    axes[0].set_title('Image')
    axes[1].imshow(mask.squeeze(), cmap='gray')
    axes[1].set_title('Binary Mask')
    plt.tight_layout()
    plt.savefig('sample_check.png')
    print("Saved sample_check.png")
