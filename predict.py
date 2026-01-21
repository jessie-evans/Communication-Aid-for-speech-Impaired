import tensorflow as tf
import numpy as np
import cv2
import pickle
import os

# Load model
model = tf.keras.models.load_model("models/asl_model.h5")

# Load labels
with open("models/labels.pkl", "rb") as f:
    class_names = pickle.load(f)

print("✅ Model & labels loaded")

# ---------- Function to predict ----------
def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        return
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Could not read image: {img_path}")
        return

    img = cv2.resize(img, (64, 64))        # resize to training size
    img = img.astype("float32") / 255.0    # normalize
    img = np.expand_dims(img, axis=0)      # add batch dimension

    predictions = model.predict(img, verbose=0)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    print(f"👉 {os.path.basename(img_path)} ➝ {predicted_class} ({confidence*100:.2f}% confidence)")

# ---------- Test with one image ----------
predict_image("dataset/asl_alphabet/asl_alphabet_test/asl_alphabet_test/A_test.jpg")

# ---------- Optional: Test ALL images in test folder ----------
print("\n🔎 Testing all images in test folder...")
test_dir = "dataset/asl_alphabet/asl_alphabet_test/asl_alphabet_test"

for file in os.listdir(test_dir):
    if file.endswith(".jpg"):
        predict_image(os.path.join(test_dir, file))
