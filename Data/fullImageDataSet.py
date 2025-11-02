import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def pointsToBounds(points):
    Xs = [point['x'] for point in points]
    Ys = [point['y'] for point in points]
    return [min(Xs), min(Ys), max(Xs), max(Ys)]

class FullImageDataSet(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None, train=True, train_ratio=0.8):
        self.image_dir = image_dir
        self.image_files = [image for image in os.listdir(image_dir) if '.jpg' in image]
        self.transform = transform
        with open(annotation_dir, 'r') as f:
            # We are only concerned with the bounds (points) for this training set
            self.annotations = {annotation['image']: annotation['points'] for annotation in json.load(f)}

        num_train = int(len(self.image_files) * train_ratio)
        if train:
            self.image_files = self.image_files[:num_train]
        else:
            self.image_files = self.image_files[num_train:]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_name)
       	image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        image_annotation = self.annotations[image_name]
        image_bounds = pointsToBounds(image_annotation)
        labels = torch.tensor([float(x) / 512 for x in image_bounds], dtype=torch.float32)

        return image, labels
