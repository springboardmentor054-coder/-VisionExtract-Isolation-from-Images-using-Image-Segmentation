import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ─────────────────────────────────────────────
# Basic Blocks
# ─────────────────────────────────────────────

class ConvBnRelu(nn.Module):
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
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────
# Attention Gate
# ─────────────────────────────────────────────

class AttentionGate(nn.Module):
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
# Main Model
# ─────────────────────────────────────────────

class ImprovedUNet(nn.Module):
    def __init__(self, pretrained=True, use_attention=False):
        super().__init__()
        self.use_attention = use_attention

        backbone = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if pretrained else None
        )

        # Encoder
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4

        # Decoder
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)

        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            ConvBnRelu(32, 32),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        # Attention gates (optional)
        if use_attention:
            self.ag4 = AttentionGate(512, 256, 128)
            self.ag3 = AttentionGate(256, 128, 64)
            self.ag2 = AttentionGate(128, 64, 32)
            self.ag1 = AttentionGate(64, 64, 32)

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)
        ep = self.pool(e0)
        e1 = self.enc1(ep)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Decoder
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

        out = self.final_up(d1)
        out = self.final_conv(out)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

        return out


# ─────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        inter = (pred * target).sum()
        return 1 - (2. * inter + self.smooth) / (pred.sum() + target.sum() + self.smooth)


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, target):
        return self.alpha * self.bce(pred, target) + (1 - self.alpha) * self.dice(pred, target)


# ─────────────────────────────────────────────
# Test Run
# ─────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ImprovedUNet(pretrained=False, use_attention=True).to(device)

    dummy = torch.randn(1, 3, 256, 256).to(device)
    out = model(dummy)

    print("Input:", dummy.shape)
    print("Output:", out.shape) model.py 