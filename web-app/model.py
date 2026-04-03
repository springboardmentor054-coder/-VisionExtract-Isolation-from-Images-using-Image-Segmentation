"""
VisionExtract - Model Module
UNet with a pretrained ResNet34 encoder for binary subject segmentation.

Architecture:
    Encoder : ResNet34 (pretrained on ImageNet) → extracts rich feature maps
    Decoder : Series of upsampling blocks with skip connections from encoder
    Head    : 1x1 Conv → sigmoid → binary mask (subject vs background)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ─────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm → ReLU block."""
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    """Two consecutive ConvBnRelu blocks (standard UNet decoder unit)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Upsample → concat skip connection → DoubleConv.
    Used in every decoder stage.
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch from odd dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────
# Attention Gate (optional, improves accuracy)
# ─────────────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Soft attention gate to focus decoder on relevant regions.
    Applied to skip connections before concatenation.
    """
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, 1, bias=False),
            nn.BatchNorm2d(f_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, 1, bias=False),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ─────────────────────────────────────────────
# Main Model: ResNet34-UNet
# ─────────────────────────────────────────────

class VisionExtractUNet(nn.Module):
    """
    UNet with ResNet34 encoder for binary subject segmentation.

    Encoder feature channels (ResNet34):
        layer0 (stem) : 64
        layer1        : 64
        layer2        : 128
        layer3        : 256
        layer4        : 512  ← bottleneck

    Args:
        pretrained : Use ImageNet pretrained ResNet34 encoder
        use_attention: Add attention gates on skip connections
    """

    def __init__(self, pretrained=True, use_attention=False):
        super().__init__()
        self.use_attention = use_attention

        # ── Encoder (ResNet34) ──────────────────────────
        backbone = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if pretrained else None
        )

        # Split ResNet into named stages for easy skip connection access
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1,
                                   backbone.relu)           # out: 64,  H/2
        self.pool  = backbone.maxpool                       # out: 64,  H/4
        self.enc1  = backbone.layer1                        # out: 64,  H/4
        self.enc2  = backbone.layer2                        # out: 128, H/8
        self.enc3  = backbone.layer3                        # out: 256, H/16
        self.enc4  = backbone.layer4                        # out: 512, H/32 (bottleneck)

        # ── Decoder ────────────────────────────────────
        self.dec4  = DecoderBlock(512, 256, 256)
        self.dec3  = DecoderBlock(256, 128, 128)
        self.dec2  = DecoderBlock(128, 64,   64)
        self.dec1  = DecoderBlock(64,  64,   64)

        # Final upsampling back to original resolution
        self.final_up   = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            ConvBnRelu(32, 32),
            nn.Conv2d(32, 1, kernel_size=1),   # → (B, 1, H, W)
        )

        # ── Optional Attention Gates ────────────────────
        if use_attention:
            self.ag4 = AttentionGate(256, 256, 128)
            self.ag3 = AttentionGate(128, 128,  64)
            self.ag2 = AttentionGate(64,   64,  32)
            self.ag1 = AttentionGate(64,   64,  32)

    def forward(self, x):
        # ── Encode ─────────────────────────────────────
        e0 = self.enc0(x)          # (B, 64,  H/2,  W/2)
        ep = self.pool(e0)         # (B, 64,  H/4,  W/4)
        e1 = self.enc1(ep)         # (B, 64,  H/4,  W/4)
        e2 = self.enc2(e1)         # (B, 128, H/8,  W/8)
        e3 = self.enc3(e2)         # (B, 256, H/16, W/16)
        e4 = self.enc4(e3)         # (B, 512, H/32, W/32)  ← bottleneck

        # ── Decode (with optional attention) ──────────
        if self.use_attention:
            d4 = self.dec4(e4, self.ag4(e4, e3))
            d3 = self.dec3(d4, self.ag3(d4, e2))
            d2 = self.dec2(d3, self.ag2(d3, e1))
            d1 = self.dec1(d2, self.ag1(d2, e0))
        else:
            d4 = self.dec4(e4, e3)
            d3 = self.dec3(d4, e2)
            d2 = self.dec2(d3, e1)
            d1 = self.dec1(d2, e0)

        # ── Final head ─────────────────────────────────
        out = self.final_up(d1)    # (B, 32, H, W)
        out = self.final_conv(out) # (B, 1,  H, W)

        # Ensure output matches input spatial size exactly
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out  # sigmoid removed, BCEWithLogitsLoss handles it internally  


# ─────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Dice loss for binary segmentation. Smooth=1 avoids division by zero."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred   = torch.sigmoid(pred)
        pred   = pred.view(-1)
        target = target.view(-1)
        inter  = (pred * target).sum()
        return 1 - (2. * inter + self.smooth) / (pred.sum() + target.sum() + self.smooth)


class CombinedLoss(nn.Module):
    """
    Dice + BCE loss combination.
    BCE handles class boundaries, Dice handles overlap quality.
    alpha controls the BCE weight (1-alpha = Dice weight).
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.bce  = nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, target):
        return self.alpha * self.bce(pred, target) + (1 - self.alpha) * self.dice(pred, target)


# ─────────────────────────────────────────────
# Model summary / quick check
# ─────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = VisionExtractUNet(pretrained=True, use_attention=True).to(device)

    dummy  = torch.randn(2, 3, 256, 256).to(device)
    out    = model(dummy)

    print(f"Input  shape : {dummy.shape}")    # (2, 3, 256, 256)
    print(f"Output shape : {out.shape}")      # (2, 1, 256, 256)
    print(f"Output range : [{out.min():.3f}, {out.max():.3f}]")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params : {total_params:,}")
