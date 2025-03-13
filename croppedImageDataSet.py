import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CroppedImageDataSet(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [image for image in os.listdir(image_dir) if '.jpg' in image]
        self.transform = transform
        with open(annotation_dir, 'r') as f:
            # We are only concerned with the class for this training set
            self.annotations = {annotation['image']: annotation['class'] for annotation in json.load(f)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        image_class = self.annotations[image_name]
        label = torch.tensor(float(image_class), dtype=torch.float32)

        return image, label
