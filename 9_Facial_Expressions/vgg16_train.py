import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from PIL import Image

# ==============================
# ✅ Safe loader (skip bad images)
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
# ✅ Paths
# ==============================
data_dir = r"C:\Users\Charantej\Desktop\Image_classification\9_Facial_Expressions"
train_dir = os.path.join(data_dir, "train_class")
val_dir = os.path.join(data_dir, "val_class")
test_dir = os.path.join(data_dir, "test_class")

# ==============================
# ✅ Data transforms
# ==============================
transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# ==============================
# ✅ Load datasets
# ==============================
datasets_dict = {
    "train": CleanImageFolder(train_dir, transform["train"]),
    "val": CleanImageFolder(val_dir, transform["val"])
}

dataloaders = {
    x: DataLoader(datasets_dict[x], batch_size=32, shuffle=True, num_workers=0)
    for x in ["train", "val"]
}

class_names = datasets_dict["train"].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"✅ Using device: {device}")
print(f"✅ Classes: {class_names}")

# ==============================
# ✅ Load pretrained VGG16
# ==============================
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
for param in model.features.parameters():
    param.requires_grad = False

num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, len(class_names)),
    nn.LogSoftmax(dim=1)
)
model = model.to(device)

# ==============================
# ✅ Loss and Optimizer
# ==============================
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# ==============================
# ✅ Training function
# ==============================
def train_model(model, criterion, optimizer, num_epochs=15):
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.upper()} Batches"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(datasets_dict[phase])
            epoch_acc = running_corrects.double() / len(datasets_dict[phase])
            print(f"{phase} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), "best_vgg16.pth")
                print("💾 Saved new best model")

    print(f"\n✅ Training complete! Best val acc: {best_acc:.4f}")
    return model

# ==============================
# ✅ Train
# ==============================
model = train_model(model, criterion, optimizer, num_epochs=15)
