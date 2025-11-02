from flask import Flask, jsonify, render_template, request, send_from_directory
import os
import json
import math
from pathlib import Path

# Directories
DEFAULT_IMAGE_FOLDER = "data/FULL_IMAGES"
DEFAULT_ANNOTATION_FOLDER = "annotations"
DEFAULT_ANNOTATION_FILENAME = "annotations.json"
IMAGE_EXTENSIONS = tuple(ext.strip().lower() for ext in os.getenv("IMAGE_EXTENSIONS", ".png,.jpg,.jpeg").split(","))

IMAGE_FOLDER = os.getenv("IMAGE_FOLDER", DEFAULT_IMAGE_FOLDER)
ANNOTATION_FOLDER = os.getenv("ANNOTATION_FOLDER", DEFAULT_ANNOTATION_FOLDER)
ANNOTATION_FILENAME = os.getenv("ANNOTATION_FILENAME", DEFAULT_ANNOTATION_FILENAME)

IMAGE_DIRECTORY = Path(IMAGE_FOLDER)
ANNOTATION_DIRECTORY = Path(ANNOTATION_FOLDER)
ANNOTATION_PATH = ANNOTATION_DIRECTORY / ANNOTATION_FILENAME

app = Flask(__name__, template_folder="website", static_folder=IMAGE_FOLDER)

# Ensure the annotation folder exists
ANNOTATION_DIRECTORY.mkdir(parents=True, exist_ok=True)

@app.route("/")
def index():
    return render_template("hand-boxer.html")  # Serve your HTML file

@app.route("/get_images")
def get_images():
    """Retrieve list of images in the folder"""
    if not IMAGE_DIRECTORY.exists():
        return jsonify([])

    images = [
        f"/{IMAGE_FOLDER}/{file.name}"
        for file in IMAGE_DIRECTORY.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]
    print(f"Images found: {images}")
    return jsonify(images)

@app.route(f"/{IMAGE_FOLDER}/<filename>")
def serve_custom_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

# Storing data as (minx, maxx, miny, maxy)
def normalize_json_points(points):
    points_list = points['points']
    x_list = [math.ceil(point['x']) for point in points_list]
    y_list = [math.ceil(point['y']) for point in points_list]
    points['points'] = {'x_min': min(x_list), 'x_max': max(x_list), 'y_min': min(y_list), 'y_max': max(y_list)}
    return points
    
@app.route("/save_annotations", methods=["POST"])
def save_annotations():
    print("Received POST request")
    """Save annotation data to a JSON file"""
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No data received"}), 400

    # Normalize the data to a square
    data = normalize_json_points(data)

    # Load existing annotations if file exists
    if ANNOTATION_PATH.exists():
        with ANNOTATION_PATH.open("r+") as f:
            try:
                annotations = json.load(f)
            except json.JSONDecodeError:
                annotations = []
    else:
        annotations = []

    # Append new annotation
    annotations.append(data)

    # Save updated annotations
    with ANNOTATION_PATH.open("w") as f:
        json.dump(annotations, f, indent=2)

    print(f"Annotations saved: {annotations}")
    return jsonify({"status": "success", "message": "Annotation saved!"})

if __name__ == "__main__":
    app.run(debug=True)
