import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# ==============================
# ✅ Safe Loader
# ==============================
def safe_loader(path):
    try:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    except:
        print(f"⚠️ Skipping bad file: {path}")
        return None

class CleanImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform, loader=safe_loader)
        self.samples = [(p, t) for p, t in self.samples if os.path.exists(p)]

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if sample is None:
            return self.__getitem__((index + 1) % len(self.samples))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

# ==============================
# ✅ Paths & Transforms
# ==============================
data_dir = r"C:\Users\Charantej\Desktop\Image_classification\9_Facial_Expressions"
test_dir = os.path.join(data_dir, "test_class")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = CleanImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
class_names = test_dataset.classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")
print(f"✅ Classes: {class_names}")

# ==============================
# ✅ Load Trained VGG16 Model
# ==============================
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, len(class_names)),
    nn.LogSoftmax(dim=1)
)
model.load_state_dict(torch.load("best_vgg16.pth", map_location=device))
model = model.to(device)
model.eval()

# ==============================
# ✅ Evaluate on Test Set
# ==============================
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# ==============================
# ✅ Classification Report
# ==============================
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ==============================
# ✅ Confusion Matrix
# ==============================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - VGG16")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.show()
