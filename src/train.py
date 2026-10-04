"""Stage 3: train the CNN, save models/model.pth and models/history.csv."""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from model import build_model


def load(split):
    x = torch.from_numpy(np.load(f"data/processed/{split}_x.npy")).float()
    y = torch.from_numpy(np.load(f"data/processed/{split}_y.npy")).long()
    return x, y


def run_epoch(model, x, y, loss_fn, dev, bs, opt=None, flip=False):
    training = opt is not None
    model.train(training)
    idx = torch.randperm(len(x)) if training else torch.arange(len(x))
    total_loss, correct = 0.0, 0
    with torch.set_grad_enabled(training):
        for i in range(0, len(x), bs):
            b = idx[i:i + bs]
            xb, yb = x[b].to(dev), y[b].to(dev)
            if training and flip:
                 mask = torch.rand(len(xb), device=dev) < 0.5
                 xb[mask] = xb[mask].flip(3)
            out = model(xb)
            loss = loss_fn(out, yb)
            if training:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * len(b)
            correct += (out.argmax(1) == yb).sum().item()
    return total_loss / len(x), correct / len(x)


def main():
    params = yaml.safe_load(open("params.yaml"))
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    torch.set_num_interop_threads(1)
    p, size = params["train"], params["preprocess"]["image_size"]
    torch.manual_seed(p["seed"])
    dev = "cpu"

    xtr, ytr = load("train")
    xva, yva = load("val")
    model = build_model(p["num_filters"], p["dropout_rate"], p["dense_units"], size).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    rows = []
    for ep in range(1, p["epochs"] + 1):
        tl, ta = run_epoch(model, xtr, ytr, loss_fn, dev, p["batch_size"], opt, p["augment_flip"])
        vl, va = run_epoch(model, xva, yva, loss_fn, dev, p["batch_size"])
        rows.append(dict(epoch=ep, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=va))
        print(f"epoch {ep:02d} | train {tl:.3f}/{ta:.3f} | val {vl:.3f}/{va:.3f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/model.pth")
    pd.DataFrame(rows).to_csv("models/history.csv", index=False)


if __name__ == "__main__":
    main()
