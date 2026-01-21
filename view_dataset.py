# view_dataset.py
import os, random, cv2
import numpy as np

DATASET_DIR = r"D:\sign-language-detector\dataset"

# e letters list nundi meeku kavalsina letters select avuthayi
letters_to_show = list("ABCDEF")  # change: e.g., list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

samples_per_letter = 6  # each letter ki entha show cheyyalo

tile_size = 96  # display clarity kosam
gap = 6         # tiles madhya gap

rows = len(letters_to_show)
cols = samples_per_letter

# grid canvas size calc
height = rows * tile_size + (rows+1) * gap
width  = cols * tile_size + (cols+1) * gap
canvas = np.full((height, width, 3), 30, dtype=np.uint8)  # dark background

font = cv2.FONT_HERSHEY_SIMPLEX

for r, letter in enumerate(letters_to_show):
    folder = os.path.join(DATASET_DIR, letter)
    if not os.path.isdir(folder):
        print(f"Folder not found for {letter}")
        continue

    # files pick
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    if not files:
        print(f"No images in {folder}")
        continue
    picks = random.sample(files, min(samples_per_letter, len(files)))

    # row title (letter)
    cv2.putText(canvas, f"{letter}", (gap, (r+1)*gap + r*tile_size - 4),
                font, 0.8, (200,200,200), 2)

    for c, fname in enumerate(picks):
        img_path = os.path.join(folder, fname)
        img = cv2.imread(img_path)
        if img is None: 
            continue
        img = cv2.resize(img, (tile_size, tile_size))
        y = (r+1)*gap + r*tile_size
        x = (c+1)*gap + c*tile_size
        canvas[y:y+tile_size, x:x+tile_size] = img

# show window
cv2.imshow("ASL Dataset Samples (press any key to close)", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
