import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
# EPOCHS = 10
EPOCHS = 15

DATASET_PATH = "data"

# ======================
# LOAD DATA FROM CSV
# ======================

def load_dataset(split):

    csv_path = os.path.join(DATASET_PATH, split, "_classes.csv")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    images = []
    labels = []

    for _, row in df.iterrows():

        img_path = os.path.join(DATASET_PATH, split, row["filename"])

        if os.path.exists(img_path):

            images.append(img_path)

            if row["cancer"] == 1:
                labels.append(1)   # malignant
            else:
                labels.append(0)   # benign

    return images, labels


train_imgs, train_labels = load_dataset("train")
val_imgs, val_labels = load_dataset("valid")

print("Train samples:", len(train_imgs))
print("Validation samples:", len(val_imgs))


# ======================
# IMAGE PREPROCESSING
# ======================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
])
def preprocess(path, label):

    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    # img = tf.image.resize(img, IMG_SIZE)
    # img = img / 255.0
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    img = data_augmentation(img)

    return img, label


train_ds = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))
train_ds = train_ds.map(preprocess).shuffle(len(train_imgs)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_imgs, val_labels))
val_ds = val_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ======================
# MODEL
# ======================

# base_model = EfficientNetB2(
#     include_top=False,
#     weights="imagenet",
#     input_shape=(224, 224, 3)
# )

# base_model.trainable = False
base_model = tf.keras.applications.EfficientNetB2(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3)
)

# Enable training
base_model.trainable = True

# Freeze early layers, train deeper ones
for layer in base_model.layers[:-20]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(base_model.input, output)

model.compile(
    # optimizer="adam",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ======================
# TRAIN
# ======================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)
# model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=EPOCHS,
#     class_weight=class_weight,
#     callbacks=[early_stop]
# )


# ======================
# SAVE MODEL
# ======================

os.makedirs("models", exist_ok=True)

model.save("models/bone_cancer_model.h5")

print("Model saved to models/bone_cancer_model.h5")