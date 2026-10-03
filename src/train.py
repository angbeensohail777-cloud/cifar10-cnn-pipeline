from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset


def read_parameters():
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)["train"]


class CifarCNN(nn.Module):
    def __init__(self, filters, dropout):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(filters, filters * 2, kernel_size=3, padding=1),
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


def load_data():
    train_data = torch.load(
        "data/processed/train.pt",
        weights_only=False
    )

    validation_data = torch.load(
        "data/processed/val.pt",
        weights_only=False
    )

    train_dataset = TensorDataset(
        train_data["images"],
        train_data["labels"]
    )

    validation_dataset = TensorDataset(
        validation_data["images"],
        validation_data["labels"]
    )

    return train_dataset, validation_dataset


def main():
    settings = read_parameters()

    filters = settings["num_filters"]
    dropout = settings["dropout_rate"]
    learning_rate = settings["learning_rate"]
    epochs = settings["epochs"]
    batch_size = settings["batch_size"]

    train_dataset, validation_dataset = load_data()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = CifarCNN(filters, dropout).to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    history = []

    for epoch in range(epochs):
        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            predictions = model(images)
            loss = loss_function(predictions, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted_labels = predictions.argmax(dim=1)

            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total

        model.eval()

        validation_correct = 0
        validation_total = 0

        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(device)
                labels = labels.to(device)

                predictions = model(images)
                predicted_labels = predictions.argmax(dim=1)

                validation_correct += (
                    predicted_labels == labels
                ).sum().item()

                validation_total += labels.size(0)

        validation_accuracy = (
            validation_correct / validation_total
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy
        })

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Loss: {train_loss:.4f} - "
            f"Train Accuracy: {train_accuracy:.4f} - "
            f"Validation Accuracy: {validation_accuracy:.4f}"
        )

    Path("models").mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        "models/model.pth"
    )

    pd.DataFrame(history).to_csv(
        "models/history.csv",
        index=False
    )

    print("\nTraining has been completed successfully.")
    print("Model has beensaved to models/model.pth")
    print("History has been saved to models/history.csv")


if __name__ == "__main__":
    main()