import sys
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

from imageToHandBoxCNN import ImageToHandBoxCNN
from handToDigitsCNN import HandToDigitsCNN


HAND_TO_DIGITS_MODEL_PATH = "models/handToDigits_model.pth"
IMAGE_TO_HAND_BOX_MODEL_PATH = "imageToHandBox_model.pth"
INPUT_IMAGE_SIZE: Tuple[int, int] = (28, 28)


# Set device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def _load_state_dict(model: torch.nn.Module, path: str) -> None:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=True)


# Set up models, load weights, set to eval mode
IMAGE_TO_HAND_BOX_MODEL = ImageToHandBoxCNN().to(device)
_load_state_dict(IMAGE_TO_HAND_BOX_MODEL, IMAGE_TO_HAND_BOX_MODEL_PATH)
IMAGE_TO_HAND_BOX_MODEL.eval()

HAND_TO_DIGITS_MODEL = HandToDigitsCNN().to(device)
_load_state_dict(HAND_TO_DIGITS_MODEL, HAND_TO_DIGITS_MODEL_PATH)
HAND_TO_DIGITS_MODEL.eval()


def model_predict(model: torch.nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
    batch = input_tensor.to(device)
    if batch.ndim == 3:
        batch = batch.unsqueeze(0)
    with torch.no_grad():
        return model(batch)


def _to_pil_image(full_image: Union[str, np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(full_image, Image.Image):
        return full_image
    if isinstance(full_image, str):
        return Image.open(full_image)
    if isinstance(full_image, np.ndarray):
        if full_image.ndim == 2:
            return Image.fromarray(full_image)
        if full_image.shape[2] == 3:
            return Image.fromarray(full_image[:, :, ::-1])
        return Image.fromarray(full_image[:, :, :3])
    raise TypeError("full_image must be a path, numpy array, or PIL.Image")


def process_full_image(full_image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
    image = _to_pil_image(full_image).convert("L")
    image = image.resize(INPUT_IMAGE_SIZE)
    tensor = ToTensor()(image)
    return tensor.unsqueeze(0)


def crop_image(
    full_image_tensor: torch.Tensor,
    top: float,
    bottom: float,
    left: float,
    right: float,
) -> torch.Tensor:
    if full_image_tensor.ndim == 3:
        full_image_tensor = full_image_tensor.unsqueeze(0)

    _, _, h, w = full_image_tensor.shape

    top_i = int(max(0, min(h - 1, top)))
    bottom_i = int(max(top_i + 1, min(h, bottom)))
    left_i = int(max(0, min(w - 1, left)))
    right_i = int(max(left_i + 1, min(w, right)))

    cropped = full_image_tensor[:, :, top_i:bottom_i, left_i:right_i]
    resized = F.interpolate(
        cropped,
        size=INPUT_IMAGE_SIZE,
        mode="bilinear",
        align_corners=False,
    )
    return resized


def predict_fingers_from_full_image(
    full_image: Union[str, np.ndarray, Image.Image]
) -> int:
    full_image_tensor = process_full_image(full_image)

    box_logits = model_predict(IMAGE_TO_HAND_BOX_MODEL, full_image_tensor).squeeze(0)
    top, bottom, left, right = box_logits[:4].tolist()

    hand_boxed_tensor = crop_image(full_image_tensor, top, bottom, left, right)

    digits_logits = model_predict(HAND_TO_DIGITS_MODEL, hand_boxed_tensor)
    final_digit_prediction = int(torch.argmax(digits_logits, dim=1).item())
    return final_digit_prediction


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict-full-image-digits.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    prediction = predict_fingers_from_full_image(image_path)
    print(f"Predicted number of fingers: {prediction}")
