import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from pycocotools.coco import COCO

from dataset import CocoSegmentationDataset, get_train_transforms, get_val_transforms
from model import UNet

# 0. Environment Setup 
os.environ['TORCH_HOME'] = r'E:\torch_cache'
os.environ['XDG_CACHE_HOME'] = r'E:\torch_cache'


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
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

    # Use num_workers=2 for Windows; pin_memory=True for faster GPU transfer.
    BATCH_SIZE = 8
    NUM_WORKERS = 2 if os.name != 'nt' else 0 # Default to 0 for Windows, but we'll try 2 if requested
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # 3. Model & Loss Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    # Optimization Setup: Use Adam with a lower LR for fine-tuning pre-trained weights.
    # Optimization Setup: Use Adam with localized LR for finer convergence.
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    scaler = GradScaler('cuda') # For Mixed Precision Training

    # 4. Checkpoint Management
    start_epoch = 0
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth") and "best" not in f]
    
    if checkpoint_files:
        # Sort and identify the latest checkpoint
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'epoch' in x else 0)
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
        
        try:
            # Load weights with a compatibility check for changed architectures
            checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            
            # Refinement Phase: Optimized for fast loss decrease in final 10 epochs.
            # Resetting LR to 5e-5 (Half of initial) to "jolt" the model.
            if start_epoch >= 100:
                print(f"Applying Refinement Phase: Optimizing LR to 5e-5 for fast convergence.")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 5e-5
                # Reset scheduler's 'best' to ensure it doesn't decay early
                scheduler.best = 0 if scheduler.mode == 'max' else float('inf')
                    
            print(f"Resuming training from Epoch {start_epoch}")
        except Exception as e:
            print(f"Starting fresh or error loading checkpoint: {e}")
            # Weights will default to pre-trained initialization from torchvision
            pass 
    
    best_iou = 0.0
    best_loss = float('inf')

    # 5. Training Loop
    num_epochs = 110
    print(f"Training initialized on {device}. Total epochs: {num_epochs}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        current_lr = optimizer.param_groups[0]['lr']
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (images, masks) in enumerate(train_bar):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # Using AMP for faster training
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = combined_loss(outputs, masks)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping to prevent exploration instability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            train_bar.set_postfix(loss=loss.item(), lr=f"{current_lr:.2e}")

        # Validation
        model.eval()
        val_loss = 0
        total_iou, total_dice, total_prec, total_rec, total_acc = 0, 0, 0, 0, 0
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for images, masks in val_bar:
                images, masks = images.to(device), masks.to(device)
                
                with torch.amp.autocast('cuda'):
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
                val_bar.set_postfix(val_loss=v_loss.item(), iou=iou)
        
        t_avg_loss = epoch_loss / len(train_loader)
        v_avg_loss = val_loss / len(val_loader)
        
        v_avg_iou = total_iou / len(val_loader)
        v_avg_dice = total_dice / len(val_loader)
        v_avg_prec = total_prec / len(val_loader)
        v_avg_rec = total_rec / len(val_loader)
        v_avg_acc = total_acc / len(val_loader)

        scheduler.step(v_avg_iou)
        
        print(f"\nSummary Epoch {epoch+1}/{num_epochs}: Train Loss: {t_avg_loss:.4f}, Val Loss: {v_avg_loss:.4f}, LR: {current_lr:.2e}")
        print(f"Metrics -> IoU: {v_avg_iou:.4f}, Dice: {v_avg_dice:.4f}, Acc: {v_avg_acc:.4f}, Prec: {v_avg_prec:.4f}, Rec: {v_avg_rec:.4f}")
        
        # Save best model based on IoU (standard for segmentation)
        if v_avg_iou > best_iou:
            best_iou = v_avg_iou
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iou': v_avg_iou,
                'loss': v_avg_loss,
            }, best_path)
            print(f"--> Saved new BEST model (IoU: {v_avg_iou:.4f}): {best_path}")

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