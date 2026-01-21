# webcam_predict_skeleton.py
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import time

MODEL_PATH = "models/asl_keypoint_model.h5"
LABELS_PATH = "models/labels.pkl"

# Load model and labels
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "rb") as f:
    class_names = pickle.load(f)

# Initialize mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def normalize_landmarks(lms):
    arr = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
    arr[:, :2] -= arr[0, :2]
    maxd = np.max(np.linalg.norm(arr[:, :2], axis=1))
    if maxd > 0:
        arr[:, :2] /= maxd
    return arr.flatten()

cap = cv2.VideoCapture(0)

sentence = ""
cursor_index = 0
last_pred = None
last_time = time.time()
last_detected_time = time.time()
pause_threshold = 2.5  # delay before auto-space
cursor_visible = True
cursor_timer = time.time()
cursor_blink_interval = 0.5
last_edit_time = time.time()  # track last edit for space delay

with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                    min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        label, conf = None, 0.0

        # --- Prediction ---
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            vec = normalize_landmarks(hand_landmarks.landmark).reshape(1, -1)
            preds = model.predict(vec, verbose=0)
            idx = int(np.argmax(preds))
            label = class_names[idx]
            conf = float(np.max(preds))

            cv2.putText(frame, f"{label} ({conf*100:.1f}%)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

            # Add letter if stable for >1 sec
            if label == last_pred and (time.time() - last_time) > 1.0:
                sentence = sentence[:cursor_index] + label + sentence[cursor_index:]
                cursor_index += 1
                last_time = time.time()
                last_edit_time = time.time()  # reset timer after edit
            last_pred = label
            last_detected_time = time.time()

        else:
            # Auto-space after long idle (no detection)
            if (time.time() - last_detected_time > pause_threshold and
                time.time() - last_edit_time > 3.0):  # wait at least 3s after last edit
                if not sentence.endswith(" "):
                    sentence = sentence[:cursor_index] + " " + sentence[cursor_index:]
                    cursor_index += 1
                    print("Auto-space added")
                    last_edit_time = time.time()
                last_detected_time = time.time()

            cv2.putText(frame, "Show hand to camera", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # --- UI DESIGN ---
        h, w, _ = frame.shape
        board_w = 450
        board = np.zeros((h, board_w, 3), dtype=np.uint8)

        cv2.putText(board, "Predicted Text:", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.rectangle(board, (20, 70), (board_w - 20, h - 50), (255,255,255), 2)

        # Cursor blinking
        if time.time() - cursor_timer > cursor_blink_interval:
            cursor_visible = not cursor_visible
            cursor_timer = time.time()

        # Insert cursor symbol
        display_sentence = sentence
        if cursor_visible:
            display_sentence = sentence[:cursor_index] + "|" + sentence[cursor_index:]

        # Wrap text for display
        max_width = 22
        wrapped = ""
        for i in range(0, len(display_sentence), max_width):
            wrapped += display_sentence[i:i + max_width] + "\n"

        y_offset = 110
        for line in wrapped.split("\n"):
            cv2.putText(board, line, (40, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            y_offset += 40
            if y_offset > h - 60:
                break

        combined = np.hstack((frame, board))
        cv2.imshow("ASL Keypoint Predictor", combined)

        # --- Keyboard Controls ---
        key = cv2.waitKeyEx(1)  # <-- IMPORTANT for arrow keys

        if key == ord('q'):
            break
        elif key == ord('c'):  # Clear
            sentence = ""
            cursor_index = 0
            last_edit_time = time.time()
        elif key == ord('d'):  # Delete char before cursor
            if cursor_index > 0:
                sentence = sentence[:cursor_index-1] + sentence[cursor_index:]
                cursor_index -= 1
                last_edit_time = time.time()
        elif key == 2424832:  # LEFT arrow
            cursor_index = max(0, cursor_index - 1)
        elif key == 2555904:  # RIGHT arrow
            cursor_index = min(len(sentence), cursor_index + 1)

cap.release()
cv2.destroyAllWindows()
