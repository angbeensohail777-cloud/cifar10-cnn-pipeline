from pathlib import Path
import torch
from torchvision.datasets import CIFAR10


def main():
    raw_dir = Path("data/raw")
    download_dir = raw_dir / "cifar10"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading CIFAR-10...")

    train_dataset = CIFAR10(
        root=download_dir,
        train=True,
        download=True
    )

    test_dataset = CIFAR10(
        root=download_dir,
        train=False,
        download=True
    )

    torch.save(
        {
            "images": torch.tensor(train_dataset.data),
            "labels": torch.tensor(train_dataset.targets),
        },
        raw_dir / "train.pt",
    )

    torch.save(
        {
            "images": torch.tensor(test_dataset.data),
            "labels": torch.tensor(test_dataset.targets),
        },
        raw_dir / "test.pt",
    )

    print("Hi Angbeen! CIFAR-10 Data Saved Successfully.")
    print(f"Training samples are: {len(train_dataset)}")
    print(f"Test samples are: {len(test_dataset)}")


if __name__ == "__main__":
    main()