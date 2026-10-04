"""Stage 2: normalize, split train/val, save tensors to data/processed/."""
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split


def to_tensor(x, size):
    x = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0  # NHWC -> NCHW, [0,1]
    if size != 32:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x


def normalize(x, mean, std):
    # NORMALIZATION STEP (Part E edits this line differently on two branches)
    return (x - mean) / std


def main():
    p = yaml.safe_load(open("params.yaml"))["preprocess"]
    os.makedirs("data/processed", exist_ok=True)

    x = np.load("data/raw/train_x.npy")
    y = np.load("data/raw/train_y.npy")
    xte = np.load("data/raw/test_x.npy")
    yte = np.load("data/raw/test_y.npy")

    xtr, xva, ytr, yva = train_test_split(
        x, y, test_size=p["val_size"], random_state=p["seed"], stratify=y
    )
    xtr, xva, xte = (to_tensor(a, p["image_size"]) for a in (xtr, xva, xte))

    # statistics from the training split only (no leakage)
    mean = xtr.mean(dim=(0, 2, 3), keepdim=True)
    std = xtr.std(dim=(0, 2, 3), keepdim=True)

    for name, t, lab in (("train", xtr, ytr), ("val", xva, yva), ("test", xte, yte)):
        arr = normalize(t, mean, std).numpy().astype(np.float16)  # float16 = half the size
        np.save(f"data/processed/{name}_x.npy", arr)
        np.save(f"data/processed/{name}_y.npy", lab)
        print(f"{name}: {arr.shape}")


if __name__ == "__main__":
    main()
