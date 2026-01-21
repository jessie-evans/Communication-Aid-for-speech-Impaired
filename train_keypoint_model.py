# train_keypoint_model.py
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle
import os

X = np.load("dataset_keypoints/X.npy")
y = np.load("dataset_keypoints/y.npy", allow_pickle=True)

# encode labels to integers
le = LabelEncoder()
y_enc = le.fit_transform(y)

# save label encoder classes (in order)
os.makedirs("models", exist_ok=True)
with open("models/labels.pkl", "wb") as f:
    pickle.dump(list(le.classes_), f)

# train test split
X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.15, random_state=42, stratify=y_enc)

input_shape = X_train.shape[1]  # 63

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_shape,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("models/asl_keypoint_model.h5", save_best_only=True, monitor="val_accuracy", mode="max"),
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True)
]

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=40, batch_size=64, callbacks=callbacks)

print("Best model saved to models/asl_keypoint_model.h5")
