import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pycocotools.coco import COCO

from dataset import CocoSegmentationDataset, get_train_transforms, get_val_transforms
from model import UNet

# 0. Environment Setup (Fix C: drive filling issue)
os.environ['TORCH_HOME'] = r'E:\torch_cache'
os.environ['XDG_CACHE_HOME'] = r'E:\torch_cache'


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

def combined_loss(inputs, targets):
    bce = nn.BCEWithLogitsLoss()(inputs, targets)
    dice = DiceLoss()(inputs, targets)
    return bce + dice

def calculate_metrics(outputs, targets):
    """Calculate IoU, Dice, Precision, Recall, and Accuracy for binary segmentation."""
    preds = (torch.sigmoid(outputs) > 0.5).float()
    targets = targets.float()
    
    # Flatten
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    
    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1 - targets_flat)).sum()
    fn = ((1 - preds_flat) * targets_flat).sum()
    tn = ((1 - preds_flat) * (1 - targets_flat)).sum()
    
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    accuracy = (tp + tn + 1e-6) / (tp + tn + fp + fn + 1e-6)
    
    return iou.item(), dice.item(), precision.item(), recall.item(), accuracy.item()

def main():
    # 1. Configuration
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    annotation_path = os.path.join(project_root, "data/annotations/instances_train2017.json")
    image_folder = os.path.join(project_root, "data/train2017")
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 2. Data Loading
    coco = COCO(annotation_path)

    full_dataset = CocoSegmentationDataset(
        coco,
        image_folder,
        transform=get_train_transforms()
    )

    # Use a larger portion of the dataset if available
    print(f"Full dataset size: {len(full_dataset)}")
    
    MAX_DATASET_SIZE = 30000
    dataset_size = min(len(full_dataset), MAX_DATASET_SIZE)
    
    if dataset_size < 1:
        print("Error: Dataset is empty. Check your data folder.")
        return

    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    print(f"Using {dataset_size} images (Train: {train_size}, Val: {val_size})")

    train_ds, val_ds = random_split(
        full_dataset, 
        [train_size, val_size, len(full_dataset) - dataset_size],
        generator=torch.Generator().manual_seed(42)
    )[:2]

    # Windows compatible DataLoader:
    # num_workers=0 is cleanest for Windows multiprocessing, pin_memory speed up GPU transfer.
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # 3. Model & Loss Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # 4. Resume from Checkpoint Logic
    start_epoch = 0
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if checkpoint_files:
        # Sort by epoch number extracted from filename (e.g., unet_epoch_5.pth)
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'epoch' in x else 0)
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
        print(f"Found checkpoint: {latest_checkpoint}. Resuming...")
        
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from Epoch {start_epoch}")
    else:
        print("No checkpoints found. Starting from scratch.")

    best_loss = float('inf')

    # 5. Training Loop
    num_epochs = 100
    print(f"Starting training loop for {num_epochs} epochs...")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        print(f"Epoch {epoch+1} started. Fetching batches...")
        for batch_idx, (images, masks) in enumerate(train_loader):
            if batch_idx == 0:
                print(f"First batch received! Image shape: {images.shape}")
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        total_iou, total_dice, total_prec, total_rec, total_acc = 0, 0, 0, 0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                v_loss = combined_loss(outputs, masks)
                val_loss += v_loss.item()
                
                # Metrics
                iou, dice, prec, rec, acc = calculate_metrics(outputs, masks)
                total_iou += iou
                total_dice += dice
                total_prec += prec
                total_rec += rec
                total_acc += acc
        
        t_avg_loss = epoch_loss / len(train_loader)
        v_avg_loss = val_loss / len(val_loader)
        
        v_avg_iou = total_iou / len(val_loader)
        v_avg_dice = total_dice / len(val_loader)
        v_avg_prec = total_prec / len(val_loader)
        v_avg_rec = total_rec / len(val_loader)
        v_avg_acc = total_acc / len(val_loader)

        scheduler.step(v_avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {t_avg_loss:.4f}, Val Loss: {v_avg_loss:.4f}")
        print(f"Metrics -> IoU: {v_avg_iou:.4f}, Dice: {v_avg_dice:.4f}, Acc: {v_avg_acc:.4f}, Prec: {v_avg_prec:.4f}, Rec: {v_avg_rec:.4f}")
        
        if v_avg_loss < best_loss:
            best_loss = v_avg_loss
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': v_avg_loss,
            }, best_path)
            print(f"Saved best model: {best_path}")

        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            save_path = os.path.join(checkpoint_dir, f"unet_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': v_avg_loss,
            }, save_path)
            print(f"Saved checkpoint: {save_path}")

    print("Training completed.")

if __name__ == '__main__':
    main()