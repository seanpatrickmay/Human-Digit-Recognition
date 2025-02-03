import numpy as np
import sys
import torch
from torchvision.transforms import ToTensor
from imageToHandBoxCNN import ImageToHandBoxCNN
from handToDigitsCNN import HandToDigitsCNN

HAND_TO_DIGITS_MODEL_PATH = "handToDigits_model.pth"
IMAGE_TO_HAND_BOX_MODEL_PATH = "imageToHandBox_model.pth"

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
IMAGE_TO_HAND_BOX_MODEL.load_state_dict(torch.load(IMAGE_TO_HAND_BOX_MODEL_PATH, weights_only=True))
IMAGE_TO_HAND_BOX_MODEL.eval()

# Second model, which gives how many fingers are raised on the smaller, hand image
HAND_TO_DIGITS_MODEL = HandToDigitsCNN().to(device)
HAND_TO_DIGITS_MODEL.load_state_dict(torch.load(HAND_TO_DIGITS_MODEL_PATH, weights_only=True))
HAND_TO_DIGITS_MODEL.eval()

# Method to get model's prediction given appropriate input
def model_predict(model, input_tensor):
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        return model(input_tensor)

# Method to process image into appropriate input for first model
# Filler for now, make sure to update crop_image if this doesn't return an image XXXXXXXXX
def process_full_image(full_image):
    return full_image

# Method to crop an image, given bounds for x and y
# Filler for now XXXXXXXXXX
def crop_image(full_image, top, bottom, left, right):
    return full_image
    

# Method which, given a full image, uses both models to predict how many fingers are held up.
def full_image_to_hand_box(full_image):
    full_image_tensor = process_full_image(full_image)
    box_prediction = model_predict(IMAGE_TO_HAND_BOX_MODEL, full_image_tensor)
    top, bottom, left, right = box_prediction[0], box_prediction[1], box_prediction[2], box_prediction[3]
    hand_boxed_image = crop_image(full_image_tensor, top, bottom, left, right)
    # Fix to make this into a tensor, if needed
    digits_predictions = model_predict(HAND_TO_DIGITS_MODEL, hand_boxed_image)
    # Need to get the max, as digits_prediction is list of confidences
    final_digit_prediction = torch.argmax(digits_predictions).item()
    return final_digit_prediction

