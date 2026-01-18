# test_fly_io.py
import requests
from medmnist import BreastMNIST
from torchvision import transforms
from PIL import Image
import io

# Load a test image from the dataset
test_dataset = BreastMNIST(split = 'test', download = True, size = 128, transform = transforms.ToTensor())
img, label = test_dataset[9]

# Convert tensor to PIL Image
img_pil = transforms.ToPILImage()(img)

# Save to bytes
img_bytes = io.BytesIO()
img_pil.save(img_bytes, format = 'PNG')
img_bytes.seek(0)

# Send to deployed API on Fly.io
url = 'https://breast-cancer-prediction.fly.dev/predict'
files = {'file': ('test_image.png', img_bytes, 'image/png')}
response = requests.post(url, files = files)

print(f"Actual label: {label.item()} ({'malignant' if label.item() == 1 else 'benign'})")
print(f"Prediction response: {response.json()}")