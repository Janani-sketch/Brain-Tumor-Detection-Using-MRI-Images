from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = load_model("best_model.keras")

# Disease labels
class_names = [
    "Glioma Tumor",
    "Meningioma Tumor",
    "Pituitary Tumor",
    "No Tumor"
]

# Image preprocessing
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            image = Image.open(file)
            processed_image = preprocess_image(image)
            preds = model.predict([processed_image, processed_image])
            prediction = class_names[np.argmax(preds)]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
