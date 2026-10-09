
from pathlib import Path
from collections import Counter
import random

from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder

DATASET_DIR = Path(r"D:\projects\agribrain\Datasets\2\original")
SEED = 42

random.seed(SEED)

dataset = ImageFolder(DATASET_DIR)
samples = dataset.samples
labels = [label for _, label in samples]

print("Classes:", dataset.classes)
print("Total images:", len(samples))
print("\nClass counts:")
for class_name, count in Counter(dataset.targets).items():
    print(f"{dataset.classes[class_name]}: {count}")

indices = list(range(len(samples)))

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=SEED,
    stratify=labels,
)

temp_labels = [labels[i] for i in temp_idx]

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_labels,
)

def show_split(name, split_indices):
    counts = Counter(labels[i] for i in split_indices)
    print(f"\n{name}: {len(split_indices)} images")
    for class_id, class_name in enumerate(dataset.classes):
        print(f"  {class_name}: {counts[class_id]}")

show_split("Training", train_idx)
show_split("Validation", val_idx)
show_split("Testing", test_idx)

print("\nDataset inspection and split successful.")
