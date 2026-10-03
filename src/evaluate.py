from pathlib import Path
import json

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset


class CifarCNN(nn.Module):
    def __init__(self, filters, dropout):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, filters, 3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(filters, filters * 2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters * 2 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10)
        )

    def forward(self, images):
        images = self.features(images)
        return self.classifier(images)


def load_settings():
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)


def main():
    settings = load_settings()["train"]

    filters = settings["num_filters"]
    dropout = settings["dropout_rate"]
    batch_size = settings["batch_size"]

    test_data = torch.load(
        "data/processed/test.pt",
        weights_only=False
    )

    test_dataset = TensorDataset(
        test_data["images"],
        test_data["labels"]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = CifarCNN(filters, dropout).to(device)

    model.load_state_dict(
        torch.load(
            "models/model.pth",
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    loss_function = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0

    actual_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            actual_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(predictions.cpu().tolist())

    test_loss = total_loss / len(test_loader)
    test_accuracy = correct / total

    matrix = confusion_matrix(
        actual_labels,
        predicted_labels
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix
    )

    display.plot()
    plt.title("CIFAR-10 Confusion Matrix")
    plt.tight_layout()

    Path("models").mkdir(parents=True, exist_ok=True)

    plt.savefig(
        "models/confusion_matrix.png",
        dpi=150
    )

    plt.close()

    results = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    }

    with open("metrics.json", "w") as file:
        json.dump(results, file, indent=4)

    print("Evaluation has been completed.")
    print(f"Test loss is: {test_loss:.4f}")
    print(f"Test accuracy is: {test_accuracy:.4f}")
    print("Confusion matrix saved to models/confusion_matrix.png")
    print("Metrics saved to metrics.json")


if __name__ == "__main__":
    main()