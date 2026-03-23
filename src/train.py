import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pycocotools.coco import COCO

from dataset import CocoSegmentationDataset, get_transforms
from model import UNet


annotation_path = "../data/annotations/instances_train2017.json"
image_folder = "../data/train2017"

coco = COCO(annotation_path)

image_transform, mask_transform = get_transforms()

dataset = CocoSegmentationDataset(
    coco,
    image_folder,
    image_transform=image_transform,
    mask_transform=mask_transform
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 3

for epoch in range(num_epochs):

    model.train()
    epoch_loss = 0

    for images, masks in dataloader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}")

torch.save(model.state_dict(), "../checkpoints/unet_model.pth")