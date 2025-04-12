import os
import gdown
import torch

MODEL_PATH = "model.pth"
MODEL_URL = "https://drive.google.com/uc?id=11lutEL0hLsG-_3eeUz1wb1jTPU7n6O1i"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("Download complete.")

def load_model():
    download_model()
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model()


from flask import Flask, request, jsonify
import io, os, base64
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import gdown

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Download model
MODEL_ID = "11lutEL0hLsG-_3eeUz1wb1jTPU7n6O1i"
MODEL_PATH = "best_model1_vgg.pth"

if not os.path.exists(MODEL_PATH):
    gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)

class VGGRegression(nn.Module):
    def __init__(self, backbone):
        super(VGGRegression, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

def load_model():
    vgg = models.vgg16(pretrained=True)
    vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, 1)
    model = VGGRegression(vgg)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

def preprocess(img):
    img = img.resize((224, 224))
    tensor = transforms.ToTensor()(img).unsqueeze(0)
    return tensor.to(device)

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
