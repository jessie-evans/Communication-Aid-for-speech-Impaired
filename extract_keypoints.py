import os
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import pickle
from tqdm import tqdm

# ---------- CONFIG ----------
# change this to your dataset root that contains A/, B/, ... folders
# 👇 here I fixed the path (with spaces in folder name)
DATASET_ROOT = Path("dataset/asl_alphabet/asl_alphabet_train")
OUT_DIR = Path("dataset_keypoints")
OUT_DIR.mkdir(exist_ok=True, parents=True)

MIN_IMAGES_PER_CLASS = 1   # used to detect class folders
SAVE_EVERY = 2000         # periodic save to disk
# ----------------------------

mp_hands = mp.solutions.hands

def find_class_dirs(root: Path):
    # return list of dirs that contain image files (jpg/png)
    classes = []
    for p in root.iterdir():
        if p.is_dir():
            # count images inside
            cnt = sum(1 for _ in p.glob("*.jpg")) + sum(1 for _ in p.glob("*.png"))
            if cnt >= MIN_IMAGES_PER_CLASS:
                classes.append(p)
            else:
                # maybe nested one deeper (handle double nesting)
                for sub in p.iterdir():
                    if sub.is_dir():
                        cnt2 = sum(1 for _ in sub.glob("*.jpg")) + sum(1 for _ in sub.glob("*.png"))
                        if cnt2 >= MIN_IMAGES_PER_CLASS:
                            classes.append(sub)
    return sorted(classes)

def normalize_landmarks(lms):
    # lms: 21 landmarks each with (x,y,z) in normalized coords
    arr = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)  # shape (21,3)
    # subtract wrist(0) -> make relative
    arr[:, :2] -= arr[0, :2]
    # scale by max distance in xy to be scale invariant
    maxd = np.max(np.linalg.norm(arr[:, :2], axis=1))
    if maxd > 0:
        arr[:, :2] /= maxd
    return arr.flatten()   # 63-d vector

def main():
    class_dirs = find_class_dirs(DATASET_ROOT)
    if not class_dirs:
        print("No class folders found in", DATASET_ROOT)
        return

    print("Found classes:", [p.name for p in class_dirs])

    X = []
    y = []
    failed = []

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        for class_dir in class_dirs:
            label = class_dir.name
            imgs = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            print(f"Processing {label} ({len(imgs)} images)")
            for img_path in tqdm(imgs):
                img = cv2.imread(str(img_path))
                if img is None:
                    failed.append((str(img_path), "imread_failed"))
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                if not results.multi_hand_landmarks:
                    failed.append((str(img_path), "no_hand"))
                    continue
                # take first hand
                hand_landmarks = results.multi_hand_landmarks[0]
                vec = normalize_landmarks(hand_landmarks.landmark)
                if np.any(np.isnan(vec)) or vec.shape[0] != 63:
                    failed.append((str(img_path), "bad_landmarks"))
                    continue
                X.append(vec)
                y.append(label)

                # periodic save
                if len(X) % SAVE_EVERY == 0:
                    np.save(OUT_DIR / "X_partial.npy", np.array(X))
                    np.save(OUT_DIR / "y_partial.npy", np.array(y))
        # final save
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=object)

    np.save(OUT_DIR / "X.npy", X)
    np.save(OUT_DIR / "y.npy", y)
    # save failed list
    with open(OUT_DIR / "failed.txt", "w", encoding="utf8") as f:
        for p, reason in failed:
            f.write(f"{p}\t{reason}\n")

    print("Saved X.npy (shape):", X.shape)
    print("Saved y.npy (shape):", y.shape)
    print("Failed count:", len(failed))

    # save unique label order
    classes = sorted(list(set(y)))
    with open(OUT_DIR / "labels.pkl", "wb") as f:
        pickle.dump(classes, f)
    print("Saved labels.pkl with classes:", classes)

if __name__ == "__main__":
    main()
