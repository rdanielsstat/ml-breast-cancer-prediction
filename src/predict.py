import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
from torchvision import transforms
import io
from pydantic import BaseModel
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_model.pth"

app = FastAPI()

# Set device
device = torch.device('cpu')

# Model architecture (same as train.py)
class BreastCNN(nn.Module):
    def __init__(self, conv_drop_rate = 0.0, fc_drop_rate = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop_conv = nn.Dropout2d(p = conv_drop_rate)
        self.drop_fc = nn.Dropout(p = fc_drop_rate)
        self.gap = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(128 * 4 * 4, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.drop_conv(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.drop_fc(x)
        x = self.fc(x)
        return x

# Response model
class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    confidence: float
    probabilities: dict

# Load model at startup
model = BreastCNN(conv_drop_rate = 0.0, fc_drop_rate = 0.3).to(device)
checkpoint = torch.load(MODEL_PATH, map_location = device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded successfully!")

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels = 1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

@app.post("/predict", response_model = PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict breast cancer classification from uploaded image.
    
    - **file**: Image file (PNG, JPG, etc.)
    
    Returns:
    - **prediction**: 0 (benign) or 1 (malignant)
    - **class_name**: Human-readable class name
    - **confidence**: Confidence score for the prediction
    - **probabilities**: Probability for each class
    """
    try:
        # Read and preprocess image
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes))
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim = 1)
            predicted_class = torch.argmax(probabilities, dim = 1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Return result
        return PredictionResponse(
            prediction = int(predicted_class),
            class_name = 'malignant' if predicted_class == 1 else 'benign',
            confidence = float(confidence),
            probabilities = {
                'benign': float(probabilities[0][0]),
                'malignant': float(probabilities[0][1])
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "model": "BreastCNN"}

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Breast Cancer Classification API",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Upload image for prediction",
            "/health": "GET - Health check",
            "/docs": "GET - Interactive API documentation"
        }
    }