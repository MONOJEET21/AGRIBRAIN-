
from pathlib import Path
import random
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ---------------- CONFIGURATION ----------------

DATA_DIR = Path(r"D:\projects\agribrain\Datasets\2\original")
OUTPUT_DIR = Path(r"D:\projects\agribrain\training\outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 192
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 0.001
SEED = 42

CHECKPOINT_PATH = OUTPUT_DIR / "sesame_cnn_4class_192_weighted.pth"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ---------------- IMAGE TRANSFORMS ----------------

train_transform = transforms.Compose([
    transforms.Resize((216, 216)),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

# Separate datasets: augment training images only.
train_base = datasets.ImageFolder(DATA_DIR, transform=train_transform)
eval_base = datasets.ImageFolder(DATA_DIR, transform=eval_transform)

classes = train_base.classes
labels = [label for _, label in train_base.samples]
indices = list(range(len(labels)))

if len(classes) != 4:
    raise ValueError(f"Expected 4 classes, found {len(classes)}")

print("Classes:", classes)
print("Total images:", len(labels))

# ---------------- REPRODUCIBLE SPLIT ----------------

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=SEED,
    stratify=labels,
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=SEED,
    stratify=[labels[i] for i in temp_idx],
)

train_dataset = Subset(train_base, train_idx)
val_dataset = Subset(eval_base, val_idx)
test_dataset = Subset(eval_base, test_idx)

generator = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
    generator=generator,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

print(
    f"Train: {len(train_dataset)} | "
    f"Validation: {len(val_dataset)} | "
    f"Test: {len(test_dataset)}"
)

# ---------------- CLASS-WEIGHTED LOSS ----------------
# Calculate weights using training labels only.

train_labels = np.array(labels)[train_idx]
class_counts = np.bincount(
    train_labels,
    minlength=len(classes),
)

class_weights = len(train_labels) / (
    len(classes) * class_counts
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=device,
)

print("\nClass weights:")
for class_name, count, weight in zip(
    classes, class_counts, class_weights.tolist()
):
    print(f"{class_name}: count={count}, weight={weight:.3f}")

# ---------------- CNN BUILT FROM SCRATCH ----------------

class SesameCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = SesameCNN(num_classes=len(classes)).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

print("\nCNN initialized from scratch.")
print("Checkpoint:", CHECKPOINT_PATH)

# ---------------- TRAINING ----------------

best_val_f1 = -1.0
best_epoch = 0
best_state = None

for epoch in range(EPOCHS):
    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, targets in train_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * targets.size(0)
        train_correct += (
            outputs.argmax(1) == targets
        ).sum().item()
        train_total += targets.size(0)

    train_loss = train_loss_sum / train_total
    train_acc = train_correct / train_total

    # Validation
    model.eval()

    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0
    val_predictions = []
    val_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)
            predictions = outputs.argmax(1)

            val_loss_sum += loss.item() * targets.size(0)
            val_correct += (
                predictions == targets
            ).sum().item()
            val_total += targets.size(0)

            val_predictions.extend(predictions.cpu().tolist())
            val_targets.extend(targets.cpu().tolist())

    val_loss = val_loss_sum / val_total
    val_acc = val_correct / val_total

    val_macro_f1 = f1_score(
        val_targets,
        val_predictions,
        labels=list(range(len(classes))),
        average="macro",
        zero_division=0,
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train acc: {train_acc:.2%} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val acc: {val_acc:.2%} | "
        f"Val macro F1: {val_macro_f1:.4f}"
    )

    # Keep the model with the best validation macro F1.
    if val_macro_f1 > best_val_f1:
        best_val_f1 = val_macro_f1
        best_epoch = epoch + 1
        best_state = copy.deepcopy(model.state_dict())

        torch.save(
            {
                "model_state_dict": best_state,
                "classes": classes,
                "image_size": IMAGE_SIZE,
                "model_type": "SesameCNN",
                "seed": SEED,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_val_f1,
                "best_val_accuracy": val_acc,
                "class_weights": class_weights.cpu(),
            },
            CHECKPOINT_PATH,
        )

        print("  -> Best checkpoint saved.")

if best_state is None:
    raise RuntimeError("Training did not produce a valid checkpoint.")

print(f"\nBest epoch: {best_epoch}")
print(f"Best validation macro F1: {best_val_f1:.4f}")

# ---------------- FINAL TEST EVALUATION ----------------

model.load_state_dict(best_state)
model.eval()

all_predictions = []
all_targets = []

with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device, non_blocking=True)

        outputs = model(images)
        predictions = outputs.argmax(1).cpu().tolist()

        all_predictions.extend(predictions)
        all_targets.extend(targets.tolist())

print("\nTest classification report:")
print(
    classification_report(
        all_targets,
        all_predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
        zero_division=0,
    )
)

print("Confusion matrix (rows=actual, columns=predicted):")
print(
    confusion_matrix(
        all_targets,
        all_predictions,
        labels=list(range(len(classes))),
    )
)

test_accuracy = np.mean(
    np.array(all_predictions) == np.array(all_targets)
)

test_macro_f1 = f1_score(
    all_targets,
    all_predictions,
    labels=list(range(len(classes))),
    average="macro",
    zero_division=0,
)

print(f"\nTest accuracy: {test_accuracy:.2%}")
print(f"Test macro F1: {test_macro_f1:.4f}")
print("Checkpoint saved to:", CHECKPOINT_PATH)
