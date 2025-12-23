import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from cnn_train import CNNModel  # import the class definition
import os

# ---------------- CONFIG ----------------
data_dir = r"C:\Users\Charantej\Desktop\Image_classification\9_Facial_Expressions"
test_dir = os.path.join(data_dir, "test_class")  # your test folder
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ---------------- LOAD TEST DATA ----------------
test_data = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
class_names = test_data.classes

print("✅ Using device:", device)
print("✅ Classes:", class_names)

# ---------------- LOAD TRAINED MODEL ----------------
model = CNNModel(num_classes=len(class_names))
model.load_state_dict(torch.load("best_cnn_model.pth", map_location=device))
model.to(device)
model.eval()

# ---------------- EVALUATION ----------------
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# ---------------- METRICS ----------------
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Facial Expression CNN")
plt.show()
