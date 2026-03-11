from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)

IMG_SIZE = (224,224)

model = tf.keras.models.load_model("models/bone_cancer_model.h5")

def preprocess(img):

    img = img.resize(IMG_SIZE)
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    img = Image.open(file).convert("RGB")
    img = preprocess(img)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        result = "Malignant"
        confidence = float(pred)
    else:
        result = "Benign"
        confidence = float(1 - pred)

    return jsonify({
        "prediction": result,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(port=5000)