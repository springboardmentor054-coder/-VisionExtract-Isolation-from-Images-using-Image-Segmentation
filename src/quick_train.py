import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pycocotools.coco import COCO
import sys

# Ensure src directory is in path
sys.path.append(os.path.dirname(__file__))

from dataset import CocoSegmentationDataset, get_train_transforms, get_val_transforms
from model import UNet

# 0. Configuration
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
annotation_path = os.path.join(project_root, "data/annotations/instances_train2017.json")
image_folder = os.path.join(project_root, "data/train2017")
checkpoint_dir = os.path.join(project_root, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# 1. Data Loading (Minimal subset for DEMO)
coco = COCO(annotation_path)
full_dataset = CocoSegmentationDataset(
    coco,
    image_folder,
    transform=get_train_transforms()
)

# MINIMAL SUBSET FOR QUICK RUN
dataset_size = 20  # Only 20 images
train_size = 16
val_size = 4

train_ds, val_ds = random_split(
    full_dataset, 
    [train_size, val_size, len(full_dataset) - dataset_size],
    generator=torch.Generator().manual_seed(42)
)[:2]

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

# 2. Model & Loss Setup
def combined_loss(inputs, targets):
    bce = nn.BCEWithLogitsLoss()(inputs, targets)
    # Simplified Dice for quick demo
    inputs = torch.sigmoid(inputs)
    intersection = (inputs * targets).sum()
    dice = (2.*intersection + 1e-6) / (inputs.sum() + targets.sum() + 1e-6)
    return bce + (1 - dice)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 3. Training Loop (1 Epoch only)
print(f"Starting QUICK DEMO training on {device}...")
epoch = 0
model.train()
epoch_loss = 0
for images, masks in train_loader:
    images, masks = images.to(device), masks.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = combined_loss(outputs, masks)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()

print(f"Training Loss: {epoch_loss/len(train_loader):.4f}")

# Save Checkpoint
save_path = os.path.join(checkpoint_dir, "demo_checkpoint.pth")
torch.save({
    'model_state_dict': model.state_dict(),
}, save_path)
print(f"Demo checkpoint saved: {save_path}")
