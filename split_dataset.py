import os, shutil
from sklearn.model_selection import train_test_split

# Path where all your images are currently stored
source_path = "C:/Users/dneer/SmartVision-AI/data/classification/all_images"

# Paths to create train/val/test splits
train_path = "C:/Users/dneer/SmartVision-AI/data/classification/train"
val_path = "C:/Users/dneer/SmartVision-AI/data/classification/val"
test_path = "C:/Users/dneer/SmartVision-AI/data/classification/test"

classes = os.listdir(source_path)

for cls in classes:
    os.makedirs(os.path.join(train_path, cls), exist_ok=True)
    os.makedirs(os.path.join(val_path, cls), exist_ok=True)
    os.makedirs(os.path.join(test_path, cls), exist_ok=True)

    images = os.listdir(os.path.join(source_path, cls))
    train, temp = train_test_split(images, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    for img in train:
        shutil.copy(os.path.join(source_path, cls, img),
                    os.path.join(train_path, cls, img))
    for img in val:
        shutil.copy(os.path.join(source_path, cls, img),
                    os.path.join(val_path, cls, img))
    for img in test:
        shutil.copy(os.path.join(source_path, cls, img),
                    os.path.join(test_path, cls, img))