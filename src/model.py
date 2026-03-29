import torch
import torch.nn as nn
from torchvision import models

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class ResNetUNet(nn.Module):
    def __init__(self, n_class=1):
        super().__init__()

        self.base_model = models.resnet34(weights='DEFAULT')
        self.base_layers = list(self.base_model.children())

        # Encoder: ResNet34 Layers
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = nn.Sequential(*self.base_layers[3:4]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1 = nn.Sequential(*self.base_layers[4]) # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = nn.Sequential(*self.base_layers[5]) # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = nn.Sequential(*self.base_layers[6]) # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = nn.Sequential(*self.base_layers[7]) # size=(N, 512, x.H/32, x.W/32)

        # Decoder blocks
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = double_conv(256 + 512, 512)
        self.conv_up2 = double_conv(128 + 512, 256)
        self.conv_up1 = double_conv(64 + 256, 128)
        self.conv_up0 = double_conv(64 + 128, 64)

        self.conv_before_final = double_conv(64, 32)
        self.conv_final = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        x_0 = self.layer0(x)
        x_0_1 = self.layer0_1x1(x_0)
        x_1 = self.layer1(x_0_1)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        # Decoder
        x = self.upsample(x_4)
        x = torch.cat([x, x_3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        x = torch.cat([x, x_2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        x = torch.cat([x, x_1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        x = torch.cat([x, x_0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = self.conv_before_final(x)
        x = self.conv_final(x)

        return x

# Keep class name UNet for compatibility with train.py if possible, 
# or update train.py to use ResNetUNet. Replacing for consistency.
class UNet(ResNetUNet):
    pass
