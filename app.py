import os
import io
import base64
from flask import Flask, request, jsonify
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
import gdown

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model download and loading
MODEL_ID = "11lutEL0hLsG-_3eeUz1wb1jTPU7n6O1i"
MODEL_PATH = "best_model1_vgg.pth"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)
        print("Download complete.")

class VGGRegression(nn.Module):
    def __init__(self, backbone):
        super(VGGRegression, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

def load_model():
    download_model()
    vgg = models.vgg16(pretrained=True)
    vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, 1)
    model = VGGRegression(vgg)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# --- Image preprocessing
def preprocess(img):
    img = img.resize((224, 224))
    tensor = transforms.ToTensor()(img).unsqueeze(0)
    return tensor.to(device)

# --- Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        img = Image.open(io.BytesIO(base64.b64decode(data['image']))).convert('RGB')
        input_tensor = preprocess(img)

        with torch.no_grad():
            prediction = model(input_tensor).cpu().numpy()[0][0]

        return jsonify({'predicted_value': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
