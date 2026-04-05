from flask import Flask, request, render_template
import base64
import torch
from segmentation_models_pytorch import Unet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os
from io import BytesIO
from PIL import Image

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = os.path.join(os.getcwd(), 'Model', 'best_unet_model.pth')

try:
    model = Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Model file not found at {model_path}. Please ensure the model is trained and saved.")
    model = None

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def denorm(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(img_tensor.device)
    return img_tensor * std + mean

def image_to_data_url(image_np):
    if image_np.dtype != np.uint8:
        image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(image_np).save(buffer, format='PNG')
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

kernel = np.ones((5, 5), np.uint8)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and model:
            try:
                image = Image.open(file.stream).convert('RGB')
                image_np = np.array(image)

                transformed = val_transform(image=image_np)
                input_tensor = transformed['image'].unsqueeze(0).to(device)

                with torch.no_grad():
                    pred = model(input_tensor)
                    pred_mask = (pred > 0.5).float().squeeze().cpu().numpy().astype(np.uint8)

                refined_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
                refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)

                denorm_img = denorm(input_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy().clip(0, 1)

                isolated = denorm_img.copy()
                isolated[refined_mask == 0] = 0

                original_url = image_to_data_url(image_np)
                isolated_url = image_to_data_url(isolated)

                return render_template('index.html', original_url=original_url, isolated_url=isolated_url)
            except Exception as e:
                return render_template('index.html', error=f"Processing failed: {str(e)}")
        elif not model:
            return render_template('index.html', error="Model not loaded. Please check model file.")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)