import time
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# 1. 📦 Transformations (resize, normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. 📂 Load Dataset (using folder structure)
train_dataset = datasets.ImageFolder('dataset_split/train', transform=transform)
val_dataset = datasets.ImageFolder('dataset_split/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 3. 🧠 Load Pretrained ResNet
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))  # adapt to your number of classes

# 4. ⚙️ Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 5. 🏋️‍♂️ Training Loop (simplified)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(30):  # you can increase epochs
    model.train()
    # ETA/logging: estimate remaining time based on average batch time
    epoch_start = time.time()
    total_batches = len(train_loader)
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        batch_start = time.time()
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # compute ETA using average time per completed batch
        elapsed = time.time() - epoch_start
        batches_done = batch_idx + 1
        avg_time_per_batch = elapsed / batches_done
        remaining_batches = max(0, total_batches - batches_done)
        eta_seconds = avg_time_per_batch * remaining_batches

        # print progress every 10 batches or on the last batch
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            mins, secs = divmod(int(eta_seconds), 60)
            hrs, mins = divmod(mins, 60)
            eta_str = f"{hrs:d}h{mins:02d}m{secs:02d}s" if hrs else f"{mins:02d}m{secs:02d}s"
            print(f"Epoch [{epoch+1}/30] Batch [{batch_idx+1}/{total_batches}] Loss: {loss.item():.4f} ETA: {eta_str}")
    # end epoch summary
    epoch_elapsed = time.time() - epoch_start
    print(f"Epoch [{epoch+1}/30] completed in {int(epoch_elapsed)}s, last Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "resnet_model.pth")
print("✅ Training complete!")
