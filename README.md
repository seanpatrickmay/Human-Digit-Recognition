# Human Digit Recognition

> A **two-stage**, **camera-ready**, and **edge-friendly** perception stack that spots hands in full-frame images and reports finger counts in real time.

## Overview
Human Digit Recognition couples two convolutional neural networks to deliver live finger-count predictions from webcam or still imagery. The inference server handles capture, hand localization, cropping, and classification in one pass, returning an integer representing raised fingers. An accompanying annotation workflow speeds up dataset creation, and notebooks document the experimentation path for both models.

The repository ships with pretrained weights, a Flask API plus browser client for live capture, and utility scripts for running inference against new images. The codebase favors clarity and modularity so each stage (detection, cropping, classification) can be iterated on independently. TODO: Publish sample dataset or data capture instructions so the training notebooks can be reproduced end-to-end.

## Features
- Real-time finger counting from webcam snapshots via a Flask-powered API and lightweight web interface.
- Two-stage PyTorch pipeline that first isolates the hand then classifies the number of raised digits for higher accuracy.
- Built-in annotation server for curating custom bounding-box datasets and reproducible training notebooks.

## Quickstart
```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision flask pillow numpy

# Run
cp .env.example .env  # customize if needed
python prediction_server.py  # visit http://127.0.0.1:5000 to use the camera UI
```

## Configuration
This project uses environment variables for flexibility and portability.
Create a `.env` file in the project root with entries like:

```bash
IMAGE_FOLDER=data/FULL_IMAGES
ANNOTATION_FOLDER=annotations
ANNOTATION_FILENAME=annotations.json
IMAGE_EXTENSIONS=.png,.jpg,.jpeg
HAND_TO_DIGITS_MODEL_PATH=models/handToDigits_model.pth
IMAGE_TO_HAND_BOX_MODEL_PATH=models/imageToHandBox_model.pth
```

Load the environment with your preferred method (e.g., `export $(grep -v '^#' .env | xargs)` or `python -m dotenv run -- python prediction_server.py`).

## Architecture
- Browser client (`website/camera-predictor.html`) captures frames, encodes them as base64, and posts them to the Flask inference endpoint.
- Stage 1 (`models/imageToHandBoxCNN.py`) predicts the hand bounding box; the server crops the frame before invoking Stage 2 (`models/handToDigitsCNN.py`) to classify raised fingers.
- Annotation server (`hand-boxer-server.py`) plus notebooks (`trainHandBoxer.ipynb`, `trainDigitClassifier.ipynb`) support data collection, augmentation, and training experiments.

## Next Steps
- Extend the detector to track multiple hands per frame and aggregate predictions across people.
- Incorporate temporal smoothing or lightweight pose estimation to stabilize predictions across video streams.
- Quantize and export the models for mobile or edge deployment with on-device inference.

## Tech Highlights
- **Type safety / clarity:** Explicit PyTorch modules with documented tensor shapes ease debugging and future refactors.
- **Maintainability:** Configuration is centralized through environment-aware constants and reusable preprocessing utilities.
- **Scalability:** Stateless Flask endpoints and decoupled model weights enable containerization and distributed serving.

## Example
```bash
# Example usage or script demonstration
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,<base64-encoded-frame>"}'
```

## License
TODO: Add license information.
