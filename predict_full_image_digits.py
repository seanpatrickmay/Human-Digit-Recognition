import numpy as np
import sys
import torch
from torchvision.transforms import ToTensor
from imageToHandBoxCNN import ImageToHandBoxCNN
from handToDigitsCNN import HandToDigitsCNN
from PIL import Image
import base64
import io
import torch
import torchvision.transforms as transforms

HAND_TO_DIGITS_MODEL_PATH = "handToDigits_model.pth"
IMAGE_TO_HAND_BOX_MODEL_PATH = "imageToHandBox_model.pth"

BOUND_X_MAX = 512
BOUND_Y_MAX = 512

# Set device
device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )

# Set up models, load weights, set to eval mode
# First model, which gives points to crop large image to get just the hand
IMAGE_TO_HAND_BOX_MODEL = ImageToHandBoxCNN().to(device) 
IMAGE_TO_HAND_BOX_MODEL.load_state_dict(torch.load(IMAGE_TO_HAND_BOX_MODEL_PATH, weights_only=False))
IMAGE_TO_HAND_BOX_MODEL.eval()

# Second model, which gives how many fingers are raised on the smaller, hand image
HAND_TO_DIGITS_MODEL = HandToDigitsCNN().to(device)
HAND_TO_DIGITS_MODEL.load_state_dict(torch.load(HAND_TO_DIGITS_MODEL_PATH, weights_only=False))
HAND_TO_DIGITS_MODEL.eval()

# Method to get model's prediction given appropriate input
def model_predict(model, input_tensor):
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        return model(input_tensor)

def normalize_bounds(bounds):
    x_min, y_min, x_max, y_max = bounds[0]
    normalized_bounds = [max(0, x_min), max(0, y_min), min(BOUND_X_MAX, x_max), min(BOUND_Y_MAX, y_max)]
    return normalized_bounds

# Given model-ready image, get the bounding box for the hand
def get_bounds(processed_image):
    transform = transforms.Compose([
        transforms.ToTensor()])

    image_tensor = transform(processed_image)

    image_tensor = image_tensor.unsqueeze(0)

    print(model_predict(IMAGE_TO_HAND_BOX_MODEL, image_tensor))
    print('BOUNDS')
    return model_predict(IMAGE_TO_HAND_BOX_MODEL, image_tensor).tolist()


# Method to process image into appropriate input for first model
# Input will be from the html/js route, so need to convert specifically from the json
# Output of this method is tensor
def process_full_image(full_image_data):
    image_data = full_image_data["image"].split(",")[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((512, 512))
    return image

# Method to crop an image, given bounds for x and y
# Filler for now XXXXXXXXXX
def crop_image(full_image, bounds):
    bounds = [bound * 512 for bound in bounds]
    print(bounds)
    cropped_image = full_image.crop(bounds)
    print(cropped_image)
    cropped_image = cropped_image.resize((256, 128))

    transform = transforms.Compose([
        transforms.ToTensor(),
        ])
    cropped_image_tensor = transform(cropped_image)
    cropped_image_tensor = cropped_image_tensor.unsqueeze(0)

    return cropped_image_tensor
    

# Method which, given a full image, uses both models to predict how many fingers are held up.
def full_image_to_digits_prediction(full_image):
    # Process image
    full_image_tensor = process_full_image(full_image)
    # Get bounds from first model
    bounds_pred = get_bounds(full_image_tensor)
    print(bounds_pred)
    # Normalize bounds
    normalized_bounds_pred = normalize_bounds(bounds_pred)
    hand_boxed_image = crop_image(full_image_tensor, normalized_bounds_pred)
    # Fix to make this into a tensor, if needed
    digits_predictions = model_predict(HAND_TO_DIGITS_MODEL, hand_boxed_image)
    print(digits_predictions)
    print("DIGIT PREDS")
    # Need to get the max, as digits_prediction is list of confidences
    final_digit_prediction = torch.argmax(digits_predictions).item()
    return final_digit_prediction
