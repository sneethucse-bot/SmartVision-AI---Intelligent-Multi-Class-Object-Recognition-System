import os
from PIL import Image
import matplotlib.pyplot as plt

BASE_DIR = "smartvision_dataset/classification/train"
classes = os.listdir(BASE_DIR)

for cls in classes[:5]:  # just show 5 classes
    cls_folder = os.path.join(BASE_DIR, cls)
    images = os.listdir(cls_folder)[:5]  # show first 5 images
    for img_file in images:
        img = Image.open(os.path.join(cls_folder, img_file))
        plt.imshow(img)
        plt.title(cls)
        plt.show()