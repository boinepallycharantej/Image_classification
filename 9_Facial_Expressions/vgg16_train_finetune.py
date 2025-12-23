import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from tqdm import tqdm
import os
from PIL import Image
import time

# =====================================================
# ✅ 1. Paths and transforms
# =====================================================
data_dir = r"C:\Users\Charantej\Desktop\Image_classification\9_Facial_Expressions"
train_dir = os.path.join(data_dir, "train_class")
val_dir = os.path.join(data_dir, "val_class")

# Transformations (same as before)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# =====================================================
# ✅ 2. Safe loader to skip bad images
# =====================================================
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
        valid_samples = []
        for path, target in self.samples:
            if os.path.exists(path):
                valid_samples.append((path, target))
        self.samples = valid_samples
        self.imgs = valid_samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if sample is None:
            return self.__getitem__((index + 1) % len(self.samples))
        if self.transform:
            sample = self.transform(sample)
        return sample, target

# =====================================================
# ✅ 3. Load datasets
# =====================================================
image_datasets = {
    'train': CleanImageFolder(train_dir, data_transforms['train']),
    'val': CleanImageFolder(val_dir, data_transforms['val'])
}

dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=0)
    for x in ['train', 'val']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"\n✅ Using device: {device}")
print(f"✅ Classes: {class_names}")

# =====================================================
# ✅ 4. Load pretrained VGG16 and your previous weights
# =====================================================
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# Replace classifier
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, len(class_names)),
    nn.LogSoftmax(dim=1)
)

# Load your trained weights
model.load_state_dict(torch.load("best_vgg16.pth", map_location=device), strict=False)
model = model.to(device)
print("\n✅ Loaded previous model weights (best_vgg16.pth)")

# =====================================================
# ✅ 5. Unfreeze last 3 convolutional blocks for fine-tuning
# =====================================================
for name, param in model.features.named_parameters():
    param.requires_grad = False
    if "24" in name or "26" in name or "28" in name:  # last 3 conv layers
        param.requires_grad = True

# =====================================================
# ✅ 6. Define optimizer, loss, and scheduler
# =====================================================
criterion = nn.NLLLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, verbose=True)

# =====================================================
# ✅ 7. Training loop
# =====================================================
def train_model(model, criterion, optimizer, scheduler, num_epochs=35):
    since = time.time()
    best_acc = 0.0
    best_wts = model.state_dict()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 20)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.upper()} Batches"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f"{phase} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_wts = model.state_dict()
                    torch.save(best_wts, "best_vgg16_finetuned.pth")
                    print("💾 Saved new best fine-tuned model")

    print(f"\n✅ Fine-tuning complete! Best val acc: {best_acc:.4f}")
    model.load_state_dict(best_wts)
    return model

# =====================================================
# ✅ 8. Run training
# =====================================================
model = train_model(model, criterion, optimizer, scheduler, num_epochs=35)
