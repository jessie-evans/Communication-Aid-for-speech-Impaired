import cv2
import os

# Path to dataset
DATASET_DIR = "dataset"
# Number of images to capture per class
NUM_IMAGES = 200

# Create dataset folders if they don't exist
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Open webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")
print("Type the letter you want to capture (A–Z).")

while True:
    # Ask user for the letter
    letter = input("Enter a letter (A-Z): ").upper()
    if len(letter) != 1 or not letter.isalpha():
        print("❌ Please enter a single letter (A-Z).")
        continue

    save_path = os.path.join(DATASET_DIR, letter)
    os.makedirs(save_path, exist_ok=True)

    print(f"📸 Capturing {NUM_IMAGES} images for '{letter}'...")
    count = 0

    while count < NUM_IMAGES:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to capture frame. Exiting...")
            break

        # Show the frame
        cv2.imshow("Collecting Data", frame)

        # Save the frame
        img_name = os.path.join(save_path, f"{letter}_{count}.jpg")
        cv2.imwrite(img_name, frame)
        count += 1

        # Press 'q' to stop early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"✅ Done capturing {count} images for '{letter}'.")

    # Ask if you want to continue
    cont = input("Do you want to capture another letter? (y/n): ").lower()
    if cont != "y":
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
