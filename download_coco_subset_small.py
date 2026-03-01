from datasets import load_dataset
import os
import requests
from PIL import Image
from io import BytesIO

# 25 classes
classes = [
    "car","truck","bus","motorcycle","bicycle","airplane",
    "person","traffic light","stop sign","bench",
    "dog","cat","horse","bird","cow","elephant",
    "bottle","cup","bowl","pizza","cake",
    "chair","couch","bed","potted plant"
]

# Create folders
base_path = "data/classification/all_images"
for cls in classes:
    os.makedirs(os.path.join(base_path, cls), exist_ok=True)

# Load COCO dataset in streaming mode
dataset = load_dataset("coco", split="train", streaming=True)

count_per_class = {cls: 0 for cls in classes}

for example in dataset:
    labels = example["objects"]["label_names"]  # list of strings
    img_url = example["image"]["url"]

    for lbl in labels:
        if lbl in classes and count_per_class[lbl] < 100:
            try:
                response = requests.get(img_url, timeout=5)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                idx = count_per_class[lbl] + 1
                img.save(os.path.join(base_path, lbl, f"{lbl}_{idx}.jpg"))
                count_per_class[lbl] += 1
            except:
                continue
    if all(v >= 100 for v in count_per_class.values()):
        break

print("Downloaded 100 images per class successfully!")