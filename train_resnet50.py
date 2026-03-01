import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

# -------------------------------
# Step 0: Config
# -------------------------------
BASE_DIR = "C:/Users/dneer/SmartVision-AI/smartvision_dataset/classification"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "val")
NUM_CLASSES = 26
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Step 1: Transforms
# -------------------------------
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# -------------------------------
# Step 2: Safe ImageFolder Loader
# -------------------------------
def safe_imagefolder(root, transform):
    if not os.path.exists(root):
        raise FileNotFoundError(f"Folder not found: {root}")
    
    contains_images = any(
        os.path.isfile(os.path.join(dp, f))
        for dp, dn, filenames in os.walk(root)
        for f in filenames
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    )
    
    if not contains_images:
        raise FileNotFoundError(f"No images found in {root}")
    
    return ImageFolder(root, transform=transform)

# -------------------------------
# Step 3: Load Datasets
# -------------------------------
train_dataset = safe_imagefolder(TRAIN_DIR, transform=data_transforms["train"])
val_dataset   = safe_imagefolder(VAL_DIR, transform=data_transforms["val"])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"✅ Datasets loaded:")
print(f" - Train: {len(train_dataset)} images")
print(f" - Val: {len(val_dataset)} images")
print(f" - Classes: {train_dataset.classes}")

# -------------------------------
# Step 4: Model Setup
# -------------------------------
model = models.resnet50(pretrained=True)
# Replace final layer
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------------------
# Step 5: Training Loop
# -------------------------------
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc  = correct / total * 100
    print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")
    
    # -------------------------------
    # Validation
    # -------------------------------
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_epoch_loss = val_loss / len(val_dataset)
    val_epoch_acc  = val_correct / val_total * 100
    print(f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.2f}%")
    print("="*60)

# -------------------------------
# Step 6: Save Model
# -------------------------------
model_save_path = os.path.join(BASE_DIR, "resnet50_classification.pth")
torch.save(model.state_dict(), model_save_path)
print(f"✅ Model saved at: {model_save_path}")