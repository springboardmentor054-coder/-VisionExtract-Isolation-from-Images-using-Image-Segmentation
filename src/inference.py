import torch
import cv2
import numpy as np
from model import UNet
import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet()
model.load_state_dict(torch.load("../checkpoints/unet_model.pth"))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.ToTensor()
])


image = cv2.imread("test.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

input_tensor = transform(image_rgb).unsqueeze(0).to(device)

with torch.no_grad():

    output = model(input_tensor)
    mask = torch.sigmoid(output)
    mask = mask.squeeze().cpu().numpy()

mask = (mask > 0.5).astype(np.uint8)

mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

isolated = image_rgb * mask[:,:,None]

cv2.imwrite("../outputs/result.png", cv2.cvtColor(isolated, cv2.COLOR_RGB2BGR))