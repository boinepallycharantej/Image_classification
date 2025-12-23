import os

# Path to your dataset
base_dir = r"C:\Users\Charantej\Desktop\Image_classification\9_Facial_Expressions"

# Class ID to name mapping (update if needed)
id_to_name = {
    "0": "angry",
    "1": "disgust",
    "2": "fear",
    "3": "happy",
    "4": "neutral",
    "5": "sad",
    "6": "surprise",
    "7": "contempt",
    "8": "other"
}

for split in ["train_class", "val_class", "test_class"]:
    split_path = os.path.join(base_dir, split)
    for class_id, class_name in id_to_name.items():
        old_path = os.path.join(split_path, class_id)
        new_path = os.path.join(split_path, class_name)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"✅ Renamed {old_path} → {new_path}")

print("\n🎉 All folders renamed successfully!")
