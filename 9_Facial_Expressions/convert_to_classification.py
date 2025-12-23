import os
import shutil
from tqdm import tqdm  # progress bar

# Base dataset directory
base_dir = r"C:\Users\Charantej\Desktop\Image_classification\9_Facial_Expressions"

# YOLO-style directories
splits = ['train', 'val', 'test']

# Supported image extensions
image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']

print("\n🚀 Starting conversion from YOLO to classification format...\n")

for split in splits:
    img_dir = os.path.join(base_dir, split, 'images')
    lbl_dir = os.path.join(base_dir, split, 'labels')
    out_dir = os.path.join(base_dir, f'{split}_class')

    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        print(f"⚠️ Skipping {split} — missing folders.")
        continue

    os.makedirs(out_dir, exist_ok=True)

    label_files = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]
    print(f"\n📁 Processing {split.upper()} dataset ({len(label_files)} label files found)...")

    total_copied = 0

    for lbl_file in tqdm(label_files, desc=f"Converting {split}", ncols=80):
        lbl_path = os.path.join(lbl_dir, lbl_file)

        # Find corresponding image
        img_name = None
        for ext in image_exts:
            candidate = lbl_file.replace('.txt', ext)
            if os.path.exists(os.path.join(img_dir, candidate)):
                img_name = candidate
                break
        if img_name is None:
            continue

        img_path = os.path.join(img_dir, img_name)
        with open(lbl_path, 'r') as f:
            line = f.readline().strip()
        if not line:
            continue

        class_id = line.split()[0]
        class_folder = os.path.join(out_dir, class_id)
        os.makedirs(class_folder, exist_ok=True)

        shutil.copy(img_path, os.path.join(class_folder, img_name))
        total_copied += 1

    print(f"✅ {split.upper()} conversion complete — {total_copied} images copied to {out_dir}")

print("\n🎉 Conversion finished! Classification folders created successfully.\n")
