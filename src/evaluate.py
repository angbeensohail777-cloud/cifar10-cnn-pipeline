"""Stage 4: test metrics -> metrics.json, confusion matrix -> plots/."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix

from model import build_model

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def main():
    params = yaml.safe_load(open("params.yaml"))
    p, size = params["train"], params["preprocess"]["image_size"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.from_numpy(np.load("data/processed/test_x.npy")).float()
    y = torch.from_numpy(np.load("data/processed/test_y.npy")).long()

    model = build_model(p["num_filters"], p["dropout_rate"], p["dense_units"], size).to(dev)
    model.load_state_dict(torch.load("models/model.pth", map_location=dev))
    model.eval()

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, preds = 0.0, []
    with torch.no_grad():
        for i in range(0, len(x), 500):
            xb, yb = x[i:i + 500].to(dev), y[i:i + 500].to(dev)
            out = model(xb)
            total_loss += loss_fn(out, yb).item()
            preds.append(out.argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    acc = float((preds == y.numpy()).mean())

    with open("metrics.json", "w") as f:
        json.dump({"test_loss": total_loss / len(x), "test_accuracy": acc}, f, indent=2)

    cm = confusion_matrix(y.numpy(), preds)
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10)); ax.set_yticklabels(CLASSES)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion Matrix (acc={acc:.3f})")
    fig.tight_layout()
    fig.savefig("plots/confusion_matrix.png", dpi=120)
    print(f"test_accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
