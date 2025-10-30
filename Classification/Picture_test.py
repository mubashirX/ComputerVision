from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

# same transform as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load model
num_classes = 8
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("resnet_model.pth", map_location='cpu'))
model.eval()

class_names = [
    'White_BTS_5Kg_promo',
    'White_Blue_Brite_Rs10',
    'White_Copper_Brite_Rs20',
    'White_Golden_BTS_Rs50',
    'White_Green_BTS_Rs99',
    'White_Greylish_Black_Brite_Rs50',
    'White_Red_BTS_Rs1kg',
    'White_Yellow_BTS_Rs10'
]

# 🔸 Convert to RGB to fix the error
img = Image.open("test_pics.jpg").convert("RGB")
img = transform(img).unsqueeze(0)

with torch.no_grad():
    outputs = model(img)
    _, predicted = torch.max(outputs, 1)
    print("✅ Predicted class:", class_names[predicted.item()])