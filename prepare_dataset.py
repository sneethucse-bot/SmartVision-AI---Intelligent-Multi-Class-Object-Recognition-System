# =====================================================
# STEP 0: IMPORTS
# =====================================================
import os
import json
import random
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

# =====================================================
# STEP 1: CONFIGURATION
# =====================================================
# 25 Selected COCO classes (with correct category IDs)
SELECTED_CLASSES = {
    'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4,
    'bus': 5, 'train': 6, 'truck': 7, 'traffic light': 9, 'stop sign': 11,
    'bench': 13, 'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17,
    'cow': 19, 'elephant': 20, 'bottle': 39, 'cup': 41, 'bowl': 45,
    'pizza': 53, 'cake': 55, 'chair': 56, 'couch': 57, 'potted plant': 58,
    'bed': 59
}

IMAGES_PER_CLASS = 100
BASE_DIR = "smartvision_dataset"

# =====================================================
# STEP 2: LOAD COCO DATASET IN STREAMING MODE
# =====================================================
print("📥 Loading COCO dataset in STREAMING mode (no download)...")
dataset = load_dataset("detection-datasets/coco", split="train", streaming=True)
print("✅ Dataset loaded in streaming mode!")

# =====================================================
# STEP 3: COLLECT IMAGES PER CLASS
# =====================================================
print("\n🔍 Starting image collection from COCO dataset stream...")
print(f"🎯 Target: {IMAGES_PER_CLASS} images per class\n")

# Initialize storage
class_images = {class_name: [] for class_name in SELECTED_CLASSES.keys()}
class_counts = {class_name: 0 for class_name in SELECTED_CLASSES.keys()}

# Counters
total_collected = 0
images_processed = 0
max_iterations = 50000  # Safety limit

print("⏳ Processing images from stream...")
print("💡 Progress updates every 100 images collected\n")

# Iterate over streamed dataset
for idx, item in enumerate(dataset):
    images_processed += 1

    # Progress update
    if images_processed % 1000 == 0:
        print(f"📊 Processed {images_processed} images | Collected {total_collected}/{len(SELECTED_CLASSES) * IMAGES_PER_CLASS}")

    # Safety check
    if images_processed >= max_iterations:
        print(f"⚠️ Reached safety limit of {max_iterations} iterations")
        break

    # Stop if all classes have enough images
    if all(count >= IMAGES_PER_CLASS for count in class_counts.values()):
        print("🎉 Successfully collected 100 images for ALL classes!")
        break

    # Get annotations
    annotations = item['objects']
    categories = annotations['category']

    # Check for target classes in this image
    for cat_id in categories:
        for class_name, class_id in SELECTED_CLASSES.items():
            if cat_id == class_id and class_counts[class_name] < IMAGES_PER_CLASS:
                class_images[class_name].append({
                    'image': item['image'],       # PIL Image object
                    'annotations': annotations,   # Annotations
                    'idx': images_processed       # For naming
                })
                class_counts[class_name] += 1
                total_collected += 1

                # Progress update every 100 collected
                if total_collected % 100 == 0:
                    print(f"✓ Collected {total_collected}/{len(SELECTED_CLASSES) * IMAGES_PER_CLASS} images")
                break  # Only count once per class

# Summary of collection
print("\n" + "="*60)
print("📊 COLLECTION COMPLETE:")
print("="*60)
print(f"Images Processed: {images_processed}")
print(f"Images Collected: {total_collected}\n")

for class_name, count in sorted(class_counts.items()):
    status = "✅" if count >= IMAGES_PER_CLASS else "⚠️"
    print(f"{status} {class_name:20s}: {count:3d} images")
print("="*60)

# =====================================================
# STEP 4: CREATE FOLDER STRUCTURE
# =====================================================
print("\n📁 Creating project folder structure...\n")

# Main folder
os.makedirs(BASE_DIR, exist_ok=True)

# Classification folders
for split in ['train', 'val', 'test']:
    for class_name in SELECTED_CLASSES.keys():
        os.makedirs(f"{BASE_DIR}/classification/{split}/{class_name}", exist_ok=True)

# Detection folders
os.makedirs(f"{BASE_DIR}/detection/images", exist_ok=True)
os.makedirs(f"{BASE_DIR}/detection/labels", exist_ok=True)

print("✅ Folder structure created successfully!\n")

# =====================================================
# STEP 5: TRAIN/VAL/TEST SPLIT
# =====================================================
print("="*70)
print("🔀 Preparing Train/Val/Test splits...")
print("📊 Split Ratio: 70% Train / 15% Val / 15% Test")
print("="*70 + "\n")

# Metadata dictionary
metadata = {
    'total_images': 0,
    'classes': {},
    'splits': {'train': 0, 'val': 0, 'test': 0}
}

# Split data
train_data, val_data, test_data = {}, {}, {}

for class_name in SELECTED_CLASSES.keys():
    all_items = class_images.get(class_name, [])
    if not all_items:
        print(f"⚠️ Warning: No images found for {class_name}")
        continue

    n = len(all_items)
    train_split = int(0.7 * n)
    val_split = int(0.85 * n)

    train_data[class_name] = all_items[:train_split]
    val_data[class_name] = all_items[train_split:val_split]
    test_data[class_name] = all_items[val_split:]

    # Update metadata
    metadata['classes'][class_name] = {
        'train': len(train_data[class_name]),
        'val': len(val_data[class_name]),
        'test': len(test_data[class_name]),
        'total': len(all_items)
    }
    metadata['splits']['train'] += len(train_data[class_name])
    metadata['splits']['val'] += len(val_data[class_name])
    metadata['splits']['test'] += len(test_data[class_name])
    metadata['total_images'] += len(all_items)

    print(f"{class_name:20s}: Train={len(train_data[class_name]):3d} | Val={len(val_data[class_name]):2d} | Test={len(test_data[class_name]):2d}")

# =====================================================
# STEP 6: SAVE CLASSIFICATION IMAGES
# =====================================================
print("\n💾 Saving Classification Images...")
classification_stats = {'train': 0, 'val': 0, 'test': 0}

for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
    print(f"\n📂 Processing {split_name.upper()} split...")
    for class_name, items in tqdm(split_data.items(), desc=f" {split_name}"):
        class_folder = f"{BASE_DIR}/classification/{split_name}/{class_name}"
        for img_idx, item in enumerate(items):
            img = item['image']
            annotations = item['annotations']
            bboxes = annotations['bbox']
            categories = annotations['category']
            class_id = SELECTED_CLASSES[class_name]

            for bbox, cat_id in zip(bboxes, categories):
                if cat_id == class_id:
                    x, y, w, h = bbox
                    try:
                        cropped_img = img.crop((x, y, x + w, y + h))
                        cropped_img = cropped_img.resize((224, 224), Image.LANCZOS)
                        img_filename = f"{class_name}_{split_name}_{img_idx:04d}.jpg"
                        img_path = os.path.join(class_folder, img_filename)
                        cropped_img.save(img_path, quality=95)
                        classification_stats[split_name] += 1
                    except Exception as e:
                        print(f"⚠️ Error: {class_name} image {img_idx}: {e}")
                        break

print("\n✅ CLASSIFICATION IMAGES SAVED!")

# =====================================================
# STEP 7: SAVE DETECTION DATA (YOLO FORMAT)
# =====================================================
print("\n📁 Saving Detection Images & YOLO Labels...")
detection_stats = {'images': 0, 'annotations': 0, 'objects': 0}

# COCO -> YOLO class mapping
coco_to_yolo = {class_id: idx for idx, class_id in enumerate(SELECTED_CLASSES.values())}

# Combine train + val for detection
all_detection_data = []
for class_name in SELECTED_CLASSES.keys():
    all_detection_data.extend(train_data.get(class_name, []))
    all_detection_data.extend(val_data.get(class_name, []))

for img_idx, item in enumerate(tqdm(all_detection_data, desc="Saving detection data")):
    img = item['image']
    img_width, img_height = img.size
    img_filename = f"image_{img_idx:06d}.jpg"
    img_path = os.path.join(f"{BASE_DIR}/detection/images", img_filename)
    img.save(img_path, quality=95)
    detection_stats['images'] += 1

    annotations = item['annotations']
    bboxes = annotations['bbox']
    categories = annotations['category']

    label_filename = f"image_{img_idx:06d}.txt"
    label_path = os.path.join(f"{BASE_DIR}/detection/labels", label_filename)

    yolo_annotations = []
    for bbox, cat_id in zip(bboxes, categories):
        if cat_id in coco_to_yolo:
            x, y, w, h = bbox
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height
            yolo_class_id = coco_to_yolo[cat_id]
            yolo_line = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            yolo_annotations.append(yolo_line)
    if yolo_annotations:
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        detection_stats['annotations'] += 1
        detection_stats['objects'] += len(yolo_annotations)

# =====================================================
# STEP 8: SAVE METADATA & YOLO CONFIG
# =====================================================
# YOLO config
yaml_content = f"""# SmartVision Dataset - YOLOv8 Configuration
path: {os.path.abspath(BASE_DIR)}/detection
train: images
val: images
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: traffic light
  9: stop sign
 10: bench
 11: bird
 12: cat
 13: dog
 14: horse
 15: cow
 16: elephant
 17: bottle
 18: cup
 19: bowl
 20: pizza
 21: cake
 22: chair
 23: couch
 24: potted plant
 25: bed
nc: 26
"""
yaml_path = f"{BASE_DIR}/detection/data.yaml"
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

# Metadata JSON
metadata['classification'] = classification_stats
metadata['detection'] = detection_stats
metadata['dataset_path'] = os.path.abspath(BASE_DIR)

metadata_path = f"{BASE_DIR}/dataset_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n🎉 DATASET SETUP COMPLETE!")
print(f"📁 Location: {os.path.abspath(BASE_DIR)}")