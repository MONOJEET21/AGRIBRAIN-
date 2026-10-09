
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Resolve the checkpoint relative to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT
    / "training"
    / "outputs"
    / "sesame_cnn_4class_best.pth"
)

IMAGE_SIZE = 128


class SesameCNN(nn.Module):
    def __init__(self, num_classes=4):
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

classes = checkpoint["classes"]
IMAGE_SIZE = checkpoint.get("image_size", IMAGE_SIZE)

if len(classes) != 4:
    raise ValueError(f"Expected 4 classes, found {len(classes)}")

model = SesameCNN(num_classes=len(classes)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


def predict_image(image: Image.Image):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    k = min(3, len(classes))
    scores, indices = torch.topk(probabilities, k=k)

    predictions = [
        {
            "disease": classes[index.item()],
            "confidence": round(score.item(), 4),
        }
        for score, index in zip(scores, indices)
    ]

    confidence = predictions[0]["confidence"]

    if confidence < 0.60:
        status = "uncertain"
    elif confidence < 0.80:
        status = "moderate_confidence"
    else:
        status = "high_confidence"

    return {
        "crop": "sesame",
        "prediction": predictions[0]["disease"],
        "confidence": confidence,
        "predictions": predictions,
        "status": status,
    }
