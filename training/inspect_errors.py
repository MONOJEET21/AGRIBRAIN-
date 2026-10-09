
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# Paths and settings
DATA_DIR = Path(r"D:\projects\agribrain\Datasets\2\original")
CHECKPOINT = Path(
    r"D:\projects\agribrain\training\outputs\sesame_cnn_4class_best.pth"
)
OUTPUT_DIR = Path(r"D:\projects\agribrain\training\error_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 128
SEED = 42
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Match the evaluation preprocessing used during training.
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
classes = dataset.classes
labels = [label for _, label in dataset.samples]
indices = list(range(len(labels)))

# Reproduce the exact split from train.py.
_, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=SEED,
    stratify=labels,
)

_, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=SEED,
    stratify=[labels[i] for i in temp_idx],
)

test_dataset = Subset(dataset, test_idx)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

# Recreate the same CNN architecture.
class SesameCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
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


# Load checkpoint. weights_only=False is for your own trusted checkpoint.
try:
    checkpoint = torch.load(
        CHECKPOINT, map_location=device, weights_only=False
    )
except TypeError:
    checkpoint = torch.load(CHECKPOINT, map_location=device)

model = SesameCNN(len(classes)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Find misclassified images.
errors = []
correct = 0
total = 0

with torch.no_grad():
    for batch_images, batch_labels in test_loader:
        outputs = model(batch_images.to(device))
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predictions = probabilities.max(dim=1)

        for j in range(len(batch_labels)):
            actual = batch_labels[j].item()
            predicted = predictions[j].item()
            sample_index = total + j

            if actual == predicted:
                correct += 1
            else:
                errors.append({
                    "dataset_index": test_idx[sample_index],
                    "path": dataset.samples[test_idx[sample_index]][0],
                    "actual": classes[actual],
                    "predicted": classes[predicted],
                    "confidence": confidence[j].item(),
                })

        total += len(batch_labels)

print(f"\nTest accuracy: {correct}/{total} = {correct / total:.2%}")
print("Misclassified images:", len(errors))

# Save a text report.
report_path = OUTPUT_DIR / "misclassified_images.txt"
with report_path.open("w", encoding="utf-8") as f:
    for i, error in enumerate(errors, start=1):
        line = (
            f"{i}. {error['path']}\n"
            f"   Actual: {error['actual']}\n"
            f"   Predicted: {error['predicted']}\n"
            f"   Model score: {error['confidence']:.2%}\n\n"
        )
        f.write(line)
        print(line, end="")

# Save contact sheets of errors in groups of 12.
# Undo normalization for display.
for start in range(0, len(errors), 12):
    group = errors[start:start + 12]
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for ax, error in zip(axes, group):
        image = plt.imread(error["path"])
        ax.imshow(image)
        ax.set_title(
            f"Actual: {error['actual']}\n"
            f"Pred: {error['predicted']} "
            f"({error['confidence']:.0%})",
            fontsize=9,
        )
        ax.axis("off")

    fig.tight_layout()
    output_file = OUTPUT_DIR / f"errors_{start + 1:03d}_{start + len(group):03d}.png"
    fig.savefig(output_file, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", output_file)

print("\nReports saved in:", OUTPUT_DIR)
