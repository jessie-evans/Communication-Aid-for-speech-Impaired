import os
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models

# ===================== Paths and parameters =====================
TRAIN_DIR = "dataset/asl_alphabet/asl_alphabet_train/asl_alphabet_train"
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "asl_model.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.pkl")

# Create models folder if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

# ===================== Load dataset =====================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# ===================== Class labels =====================
class_names = train_ds.class_names
print("✅ Classes found:", class_names)

# ===================== Normalize pixel values =====================
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# ===================== Model architecture =====================
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

# ===================== Compile model =====================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ===================== Train model =====================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ===================== Save model and labels =====================
model.save(MODEL_PATH)
with open(LABELS_PATH, "wb") as f:
    pickle.dump(class_names, f)

print(f"✅ Training complete. Model saved as {MODEL_PATH}")
print(f"✅ Labels saved as {LABELS_PATH}")
