


"""
VisionExtract - Training Module
Full training loop with:
  - IoU, Dice, Pixel Accuracy tracking
  - Learning rate scheduling
  - Early stopping
  - Best model checkpointing
  - TensorBoard logging

Usage:
    python train.py
    python train.py --coco_root /data/coco --epochs 30 --batch_size 8
"""

import os
import argparse
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloaders
from model import VisionExtractUNet, CombinedLoss


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_metrics(pred_prob, target, threshold=0.5):
    """
    Compute IoU, Dice coefficient, and pixel accuracy.

    Args:
        pred_prob : Model output probabilities  (B, 1, H, W) float in [0,1]
        target    : Ground-truth binary masks   (B, 1, H, W) float 0/1
        threshold : Binarization threshold

    Returns:
        dict with keys: iou, dice, pixel_acc
    """
    pred = (pred_prob > threshold).float()

    # Flatten for metric computation
    pred_flat   = pred.view(-1)
    target_flat = target.view(-1)

    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum()

    smooth = 1e-6
    iou       = (tp + smooth) / (tp + fp + fn + smooth)
    dice      = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    pixel_acc = (tp + tn) / (tp + fp + fn + tn + smooth)

    return {
        'iou'      : iou.item(),
        'dice'     : dice.item(),
        'pixel_acc': pixel_acc.item()
    }


# ─────────────────────────────────────────────
# Train / Validate One Epoch
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    """Run one training epoch with mixed precision."""
    model.train()
    total_loss = 0
    total_iou  = 0
    total_dice = 0
    total_acc  = 0

    pbar = tqdm(loader, desc='  Train', leave=False, unit='batch')
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            preds = model(images)
            loss  = criterion(preds, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        metrics = compute_metrics(preds.detach(), masks)
        total_loss += loss.item()
        total_iou  += metrics['iou']
        total_dice += metrics['dice']
        total_acc  += metrics['pixel_acc']

        pbar.set_postfix({'loss': f"{loss.item():.4f}",
                          'iou':  f"{metrics['iou']:.4f}"})

    n = len(loader)
    return {
        'loss'     : total_loss / n,
        'iou'      : total_iou  / n,
        'dice'     : total_dice / n,
        'pixel_acc': total_acc  / n,
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run one validation epoch."""
    model.eval()
    total_loss = 0
    total_iou  = 0
    total_dice = 0
    total_acc  = 0

    pbar = tqdm(loader, desc='  Val  ', leave=False, unit='batch')
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            preds = model(images)
            loss  = criterion(preds, masks)

        metrics = compute_metrics(preds, masks)
        total_loss += loss.item()
        total_iou  += metrics['iou']
        total_dice += metrics['dice']
        total_acc  += metrics['pixel_acc']

    n = len(loader)
    return {
        'loss'     : total_loss / n,
        'iou'      : total_iou  / n,
        'dice'     : total_dice / n,
        'pixel_acc': total_acc  / n,
    }


# ─────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────

class EarlyStopping:
    """Stop training if validation metric doesn't improve for `patience` epochs."""
    def __init__(self, patience=7, min_delta=0.001, mode='max'):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.best      = None
        self.counter   = 0
        self.triggered = False

    def step(self, metric):
        if self.best is None:
            self.best = metric
        elif self._improved(metric):
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True

    def _improved(self, metric):
        if self.mode == 'max':
            return metric > self.best + self.min_delta
        return metric < self.best - self.min_delta


# ─────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────

def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  VisionExtract Training")
    print(f"  Device     : {device}")
    print(f"  COCO root  : {cfg.coco_root}")
    print(f"  Image size : {cfg.img_size}")
    print(f"  Batch size : {cfg.batch_size}")
    print(f"  Epochs     : {cfg.epochs}")
    print(f"  Attention  : {cfg.attention}")
    print(f"{'='*60}\n")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # ── Data ───────────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        root_dir    = cfg.coco_root,
        img_size    = cfg.img_size,
        batch_size  = cfg.batch_size,
        num_workers = cfg.num_workers,
        max_train   = cfg.max_train,
        max_val     = cfg.max_val,
    )

    # ── Model, loss, optimizer ─────────────────────
    model     = VisionExtractUNet(pretrained=True,
                                  use_attention=cfg.attention).to(device)
    criterion = CombinedLoss(alpha=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    # Cosine annealing: smoothly decays LR to near-zero
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-6
    )

    scaler      = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    stopper     = EarlyStopping(patience=cfg.patience, mode='max')
    writer      = SummaryWriter(log_dir=cfg.log_dir)
    best_iou    = 0.0
    history     = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch [{epoch:03d}/{cfg.epochs}]  lr={scheduler.get_last_lr()[0]:.2e}")

        train_metrics = train_one_epoch(model, train_loader, optimizer,
                                        criterion, device, scaler)
        val_metrics   = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"  Train → loss:{train_metrics['loss']:.4f}  "
              f"IoU:{train_metrics['iou']:.4f}  "
              f"Dice:{train_metrics['dice']:.4f}  "
              f"Acc:{train_metrics['pixel_acc']:.4f}")
        print(f"  Val   → loss:{val_metrics['loss']:.4f}  "
              f"IoU:{val_metrics['iou']:.4f}  "
              f"Dice:{val_metrics['dice']:.4f}  "
              f"Acc:{val_metrics['pixel_acc']:.4f}  "
              f"[{elapsed:.1f}s]")

        # TensorBoard
        for k, v in train_metrics.items():
            writer.add_scalar(f'train/{k}', v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'val/{k}', v, epoch)
        writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

        # Save best model
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            ckpt_path = os.path.join(cfg.checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'best_iou'   : best_iou,
                'cfg'        : vars(cfg),
            }, ckpt_path)
            print(f"  ✓ New best IoU: {best_iou:.4f} — saved to {ckpt_path}")

        # Save latest checkpoint (for resuming)
        torch.save({
            'epoch'      : epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'scheduler'  : scheduler.state_dict(),
        }, os.path.join(cfg.checkpoint_dir, 'latest.pth'))

        history.append({'epoch': epoch, **train_metrics,
                        **{f'val_{k}': v for k, v in val_metrics.items()}})

        stopper.step(val_metrics['iou'])
        if stopper.triggered:
            print(f"\n  Early stopping triggered at epoch {epoch}.")
            break

    writer.close()
    print(f"\nTraining complete. Best validation IoU: {best_iou:.4f}")
    return history


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='VisionExtract Training')
    parser.add_argument('--coco_root',      type=str,   default='/path/to/your/coco',
                        help='Path to COCO 2017 dataset root')
    parser.add_argument('--img_size',       type=int,   default=256)
    parser.add_argument('--batch_size',     type=int,   default=16)
    parser.add_argument('--epochs',         type=int,   default=30)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--weight_decay',   type=float, default=1e-4)
    parser.add_argument('--num_workers',    type=int,   default=4)
    parser.add_argument('--patience',       type=int,   default=7,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--attention',      action='store_true',
                        help='Enable attention gates in decoder')
    parser.add_argument('--checkpoint_dir', type=str,   default='checkpoints')
    parser.add_argument('--log_dir',        type=str,   default='runs/visionextract')
    parser.add_argument('--max_train',      type=int,   default=None,
                        help='Limit training samples (for quick experiments)')
    parser.add_argument('--max_val',        type=int,   default=None)
    return parser.parse_args()


if __name__ == '__main__':
    cfg = parse_args()
    train(cfg)
