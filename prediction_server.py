from flask import Flask, request, jsonify, render_template
import base64
import numpy as np
from PIL import Image
import io
from predict_full_image_digits import full_image_to_digits_prediction

app = Flask(__name__, template_folder="website")

@app.route("/")
def index():
    return render_template("camera-predictor.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    predicted_class = full_image_to_digits_prediction(data)
    return jsonify({"prediction": int(predicted_class)})

if __name__ == "__main__":
    app.run(debug=True)
