import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (224,224)

model = tf.keras.models.load_model("models/bone_cancer_model.h5")

df = pd.read_csv("data/test/_classes.csv")
df.columns = df.columns.str.strip()

images = []
labels = []

for _, row in df.iterrows():

    img_path = "data/test/" + row["filename"]

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0

    images.append(img)
    labels.append(row["cancer"])

X = np.array(images)
y_true = np.array(labels)

pred = model.predict(X)
y_pred = (pred > 0.5).astype(int)

print("\nConfusion Matrix")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report")
print(classification_report(y_true, y_pred))