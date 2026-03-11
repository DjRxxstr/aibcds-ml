import tensorflow as tf
import numpy as np

IMG_SIZE = (224,224)

model = tf.keras.models.load_model("models/bone_cancer_model.h5")

def predict_image(img_path):

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        print("Prediction: MALIGNANT")
        print("Confidence:", pred)
    else:
        print("Prediction: BENIGN")
        print("Confidence:", 1 - pred)


predict_image("sample_no_cancer2.jpg")