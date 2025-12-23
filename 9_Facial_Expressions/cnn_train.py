import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
from tqdm import tqdm

# ---------------- SAFE IMAGE LOADER ----------------
def safe_loader(path):
    """Safely load images and skip missing/corrupted ones."""
    try:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    except Exception as e:
        print(f"🚫 Skipping bad or missing file: {path}")
        return None

# ---------------- SAFE IMAGE FOLDER CLASS ----------------
from torchvision.datasets import ImageFolder

class SafeImageFolder(ImageFolder):
    """Custom ImageFolder that skips corrupted images safely."""
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        # If image is None (missing/corrupted), skip to next
        if sample is None:
            print(f"⚠️ Bad image detected, skipping index {index}")
            return self.__getitem__((index + 1) % len(self.samples))

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

# ---------------- CNN MODEL ----------------
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# ---------------- TRAIN FUNCTION ----------------
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, device):
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("----------------------------------------")
        model.train()

        running_loss = 0.0
        running_corrects = 0
        total = 0

        # -------- TRAINING LOOP --------
        for images, labels in tqdm(train_loader, desc="Training Batches"):
            if images is None:
                continue  # skip bad images

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = 100 * running_corrects.double() / total

        # -------- VALIDATION LOOP --------
        model.eval()
        val_corrects = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                if images is None:
                    continue
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                val_total += labels.size(0)

        val_acc = 100 * val_corrects.double() / val_total

        print(f"Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_cnn_model.pth")
            print("✅ Saved Best Model")

    print(f"\n🎯 Training Complete! Best Val Accuracy: {best_acc:.2f}%")

# ---------------- MAIN SCRIPT ----------------
if __name__ == "__main__":
    data_dir = r"C:\Users\Charantej\Desktop\Image_classification\9_Facial_Expressions"
    train_dir = os.path.join(data_dir, "train_class")
    val_dir = os.path.join(data_dir, "val_class")  # ✅ your folder name

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("✅ Using device:", device)

    # Load datasets safely
    train_data = SafeImageFolder(train_dir, transform=transform, loader=safe_loader)
    val_data = SafeImageFolder(val_dir, transform=transform, loader=safe_loader)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    print("✅ Classes:", train_data.classes)

    # Model setup
    model = CNNModel(num_classes=len(train_data.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=30, device=device)
