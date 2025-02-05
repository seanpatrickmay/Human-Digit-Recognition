import os
import json
import cv2
import numpy as np
import albumentations as A
from copy import deepcopy
# This python script is to augment the original full pictures I have
# The purpose is to artificially increase the data set size.

PATH_TO_IMAGES = "data/FULL_IMAGES"
PATH_TO_ANNOTATIONS = "annotations/annotations.json"
OUTPUT_PATH = "data/AUGMENTED_IMAGES"

def load_annotations(path):
    with open(path, 'r') as f:
        annotations = json.load(f)
    return annotations

def save_annotations(annotations, path):
    with open(path, 'w') as f:
        json.dump(annotations, f, indent=4)

def points_to_bounds(points):
    xs = [point['x'] for point in points]
    ys = [point['y'] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]

def bounds_to_points(bounds):
    xmin, ymin, xmax, ymax = bounds
    return [
            {'x': xmin, 'y': ymin},
            {'x': xmax, 'y': ymin},
            {'x': xmin, 'y': ymax},
            {'x': xmax, 'y': ymax}
            ]

def transform_image_and_annotation(image, bounds, transform):
    transformed = transform(image=image, bboxes=[bounds], class_labels=['hand'])
    return transformed['image'], transformed['bboxes'][0]

def augment_dataset():
    os.makedirs(OUTPUT_PATH, exist_ok=True) 

    flipTransform = A.Compose([
        A.HorizontalFlip(p=1)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    rotateScaleTransform = A.Compose([
        A.Rotate(limit=15, p=1),
        A.RandomScale(scale_limit=0.2, p=1)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    brightnessContrastSaturationTransform = A.Compose([
        A.RandomBrightnessContrast(p=1),
        A.RGBShift(p=1)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    noiseTransform = A.Compose([
        A.GaussNoise(var_limit=(10, 50), p=1)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    augmented_annotations = []
    annotations = load_annotations(PATH_TO_ANNOTATIONS)

    for annotation in annotations:
        image_path = os.path.join(PATH_TO_IMAGES, annotation['image'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bounds = points_to_bounds(annotation['points'])

        for flip in [False, True]:
            if flip:
                flipped_image, flipped_bounds = transform_image_and_annotation(image, bounds, flipTransform)
            else:
                flipped_image, flipped_bounds = image, bounds

            for rotateScale in [False, True]:
                if rotateScale:
                    rotated_scaled_image, rotated_scaled_bounds = transform_image_and_annotation(flipped_image, flipped_bounds, rotateScaleTransform)
                else:
                    rotated_scaled_image, rotated_scaled_bounds = flipped_image, flipped_bounds

                for brightnessContrastSaturation in [False, True]:
                    if brightnessContrastSaturation:
                        BCS_image, BCS_bounds = transform_image_and_annotation(rotated_scaled_image, rotated_scaled_bounds, brightnessContrastSaturationTransform)
                    else:
                        BCS_image, BCS_bounds = rotated_scaled_image, rotated_scaled_bounds

                    for noise in range(3):
                        final_image, final_bounds = transform_image_and_annotation(BCS_image, BCS_bounds, noiseTransform)

                        output_image_name = f"{annotation['image']}_aug_{len(augmented_annotations)}.jpg"
                        output_image_path = os.path.join(OUTPUT_PATH, output_image_name)
                        cv2.imwrite(output_image_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

                        augmented_annotation = deepcopy(annotation)
                        augmented_annotation['image'] = output_image_name
                        augmented_annotation['points'] = bounds_to_points(final_bounds)
                        augmented_annotations.append(augmented_annotation)

    save_annotations(augmented_annotations, os.path.join(OUTPUT_PATH, 'augmented_annotations.json'))

if __name__ == '__main__':
    augment_dataset()
