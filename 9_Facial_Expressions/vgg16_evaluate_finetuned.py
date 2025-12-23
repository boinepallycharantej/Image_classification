import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import os
import pandas as pd

# ===========================================
# ✅ 1. Setup Paths
# ===========================================
data_dir = r"C:\Users\Charantej\Desktop\Image_classification\9_Facial_Expressions"
test_dir = os.path.join(data_dir, "test_class")

# ===========================================
# ✅ 2. Emotion Labels (Target 9)
# ===========================================
emotion_labels = [
    'angry', 'contempt', 'disgust', 'fear',
    'happy', 'neutral', 'other', 'sad', 'surprise'
]

# ===========================================
# ✅ 3. Safe Image Loader
# ===========================================
def safe_loader(path):
    try:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    except Exception:
        print(f"⚠️ Skipping bad image: {path}")
        return None

class CleanImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform, loader=safe_loader)
        valid_samples = [(p, t) for (p, t) in self.samples if os.path.exists(p)]
        self.samples = valid_samples
        self.imgs = valid_samples

# ===========================================
# ✅ 4. Transformations
# ===========================================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===========================================
# ✅ 5. Load Test Data (Clean 9-class mapping)
# ===========================================
test_data_raw = CleanImageFolder(test_dir, transform=transform)

valid_samples = []
for path, target in test_data_raw.samples:
    class_name = test_data_raw.classes[target]
    if class_name in emotion_labels:
        new_target = emotion_labels.index(class_name)  # Remap to 0–8
        valid_samples.append((path, new_target))

test_data_raw.samples = valid_samples
test_data_raw.targets = [t for (_, t) in valid_samples]
test_data_raw.classes = emotion_labels
test_data_raw.class_to_idx = {cls: i for i, cls in enumerate(emotion_labels)}

test_loader = torch.utils.data.DataLoader(test_data_raw, batch_size=32, shuffle=False, num_workers=0)

# ===========================================
# ✅ 6. Load Model (skip mismatched layers)
# ===========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.vgg16(weights=None)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, len(emotion_labels)),
    nn.LogSoftmax(dim=1)
)

# --- Load pretrained weights safely
checkpoint = torch.load("best_vgg16_finetuned.pth", map_location=device, weights_only=False)
model_dict = model.state_dict()
filtered_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(filtered_dict)
model.load_state_dict(model_dict, strict=False)

model.to(device)
model.eval()

print(f"\n✅ Using device: {device}")
print(f"✅ Evaluating only on classes: {emotion_labels}")

# ===========================================
# ✅ 7. Evaluate on Test Set
# ===========================================
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Evaluating Test Data"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ===========================================
# ✅ 8. Classification Report
# ===========================================
print("\n📊 Classification Report:")
report = classification_report(all_labels, all_preds, target_names=emotion_labels, zero_division=0)
print(report)

# ===========================================
# ✅ 9. Confusion Matrix
# ===========================================
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.title("Confusion Matrix - VGG16 Fine-tuned (9 Emotions)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.show()

# ===========================================
# ✅ 10. Normalized Confusion Matrix
# ===========================================
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, None]

plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens',
            xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.title("Normalized Confusion Matrix - VGG16 Fine-tuned")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.show()

# ===========================================
# ✅ 11. Accuracy Summary Table
# ===========================================
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

summary_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision (Weighted)", "Recall (Weighted)", "F1-Score (Weighted)"],
    "Score": [accuracy, precision, recall, f1]
})

print("\n✅ Overall Performance Summary:")
print(summary_df.to_string(index=False))
