from flask import Flask, jsonify, render_template, request, send_from_directory
import os
import json
import math

app = Flask(__name__, template_folder="website", static_folder="data/FULL_IMAGES")

# Directories
IMAGE_FOLDER = "data/FULL_IMAGES"
ANNOTATION_FOLDER = "annotations"

# Ensure the annotation folder exists
os.makedirs(ANNOTATION_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("hand-boxer.html")  # Serve your HTML file

@app.route("/get_images")
def get_images():
    """Retrieve list of images in the folder"""
    images = [f"/{IMAGE_FOLDER}/{file}" for file in os.listdir(IMAGE_FOLDER) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
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

    # Define the save path
    save_path = os.path.join(ANNOTATION_FOLDER, "annotations.json")

    # Load existing annotations if file exists
    if os.path.exists(save_path):
        with open(save_path, "r+") as f:
            try:
                annotations = json.load(f)
            except json.JSONDecodeError:
                annotations = []
    else:
        annotations = []

    # Append new annotation
    annotations.append(data)

    # Save updated annotations
    with open(save_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"Annotations saved: {annotations}")
    return jsonify({"status": "success", "message": "Annotation saved!"})

if __name__ == "__main__":
    app.run(debug=True)
