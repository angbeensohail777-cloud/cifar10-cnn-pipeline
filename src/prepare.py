"""Stage 1: download CIFAR-10 and save raw arrays to data/raw/."""
import os
import tempfile

import numpy as np
from torchvision.datasets import CIFAR10


def main():
    os.makedirs("data/raw", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:  # keep the tar.gz out of data/raw
        for split, is_train in (("train", True), ("test", False)):
            ds = CIFAR10(root=tmp, train=is_train, download=True)
            np.save(f"data/raw/{split}_x.npy", ds.data)  # (N, 32, 32, 3) uint8
            np.save(f"data/raw/{split}_y.npy", np.array(ds.targets, dtype=np.int64))
            print(f"{split}: {ds.data.shape}")


if __name__ == "__main__":
    main()
