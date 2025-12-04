# Human-Digit-Recognition

Creating models for recognition of human hands and fingers using personal images, plus a small web UI for building training data.

No external data used. Personal pictures used. Personal methods for creating training data.

## Project overview

- Goal: given a picture of a person’s upper body, predict how many fingers they are holding up.
- Pipeline:
  - CNN 1 (`ImageToHandBoxCNN`): take a full image and predict a bounding box around the hand.
  - CNN 2 (`HandToDigitsCNN`): take the cropped hand image and predict the number of raised fingers.

## Repository layout

- `Data/`
  - `ORIGINAL_IMAGES/`: raw photos.
  - `FULL_IMAGES/`: resized photos (500×500) used for annotation and augmentation.
  - `AUGMENTED_IMAGES/`: augmented full images and `augmented_annotations.json`.
  - `CROPPED_IMAGES/`: cropped hand images and `cropped_annotations.json`.
- `annotations/annotations.json`: bounding boxes and labels collected from the web UI.
- `hand-boxer-server.py`: Flask server that serves the annotation UI and saves annotations.
- `website/hand-boxer.html`: front-end for drawing hand bounding boxes and selecting digit labels.
- `resize-images.py`: resize raw images into `Data/FULL_IMAGES`.
- `data-augmentation.py`: augment full images and crop hand regions.
- `handToDigitsCNN.py`: digit classification model (`HandToDigitsCNN`).
- `imageToHandBoxCNN.py`: hand localization model (`ImageToHandBoxCNN`).
- `models/handToDigits_model.pth`: trained weights for the digit model.
- `imageToHandBox_model.pth`: trained weights for the hand localization model.
- `predict-full-image-digits.py`: end-to-end inference from a full image to a digit prediction.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Make sure you have a PyTorch build that matches your system (CPU / CUDA / MPS).

## Typical workflow

1. **Prepare images**
   - Place raw images into `Data/ORIGINAL_IMAGES/`.
   - Run `resize-images.py` to create 500×500 copies in `Data/FULL_IMAGES/`:
     ```bash
     python resize-images.py
     ```

2. **Annotate hands**
   - Start the Flask annotation server:
     ```bash
     python hand-boxer-server.py
     ```
   - Open `http://127.0.0.1:5000/` in a browser.
   - For each image:
     - Click 4 points around the hand (bounding box corners) on the canvas.
     - Select the digit (0–5) using the buttons.
     - Click **Next** to save the annotation.
   - Annotations are stored in `annotations/annotations.json`.

3. **Augment and crop**
   - Run the augmentation and cropping script:
     ```bash
     python data-augmentation.py
     ```
   - This will:
     - Create augmented full images and annotations in `Data/AUGMENTED_IMAGES/`.
     - Crop hand regions into `Data/CROPPED_IMAGES/` with labels in `cropped_annotations.json`.

4. **Train models (external scripts)**
   - Use the augmented full images + bounding boxes to train `ImageToHandBoxCNN`.
   - Use the cropped hand images + labels to train `HandToDigitsCNN`.
   - Save the weights to:
     - `imageToHandBox_model.pth`
     - `models/handToDigits_model.pth`

5. **Run end-to-end prediction**
   - Once both models are trained and weights are in place, run:
     ```bash
     python predict-full-image-digits.py path/to/your/image.jpg
     ```
   - The script will:
     - Preprocess the full image.
     - Use `ImageToHandBoxCNN` to predict a hand bounding box.
     - Crop and resize the hand region.
     - Use `HandToDigitsCNN` to predict the number of raised fingers.
     - Print the predicted digit.

## Notes

- Paths in the code assume the `Data/` directory spelling (capital `D`).
- The current inference pipeline assumes grayscale 28×28 inputs for both CNNs.
- You may need to retrain models if you significantly change architectures or input preprocessing.
