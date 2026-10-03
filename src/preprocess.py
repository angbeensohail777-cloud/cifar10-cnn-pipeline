from pathlib import Path

import torch
import yaml
from sklearn.model_selection import train_test_split


def load_settings():
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)


def standardize(images):
    channel_mean = torch.tensor([0.4914, 0.4822, 0.4465])
    channel_std = torch.tensor([0.2023, 0.1994, 0.2010])

    channel_mean = channel_mean.reshape(1, 3, 1, 1)
    channel_std = channel_std.reshape(1, 3, 1, 1)

    images = images.float() / 255.0
    images = images.permute(0, 3, 1, 2)

    return (images - channel_mean) / channel_std


def main():
    settings = load_settings()["preprocess"]

    validation_ratio = settings["val_size"]
    random_seed = settings["seed"]

    raw_path = Path("data/raw")
    output_path = Path("data/processed")
    output_path.mkdir(parents=True, exist_ok=True)

    train_set = torch.load(
        raw_path / "train.pt",
        weights_only=False
    )

    test_set = torch.load(
        raw_path / "test.pt",
        weights_only=False
    )

    images = standardize(train_set["images"])
    labels = train_set["labels"]

    test_images = standardize(test_set["images"])
    test_labels = test_set["labels"]

    all_indices = list(range(len(labels)))

    train_indices, validation_indices = train_test_split(
        all_indices,
        test_size=validation_ratio,
        random_state=random_seed,
        stratify=labels.numpy()
    )

    train_indices = torch.tensor(train_indices)
    validation_indices = torch.tensor(validation_indices)

    processed_train = {
        "images": images[train_indices],
        "labels": labels[train_indices]
    }

    processed_validation = {
        "images": images[validation_indices],
        "labels": labels[validation_indices]
    }

    processed_test = {
        "images": test_images,
        "labels": test_labels
    }

    torch.save(processed_train, output_path / "train.pt")
    torch.save(processed_validation, output_path / "val.pt")
    torch.save(processed_test, output_path / "test.pt")

    print("Preprocessing has been completed.")
    print("Training samples are:", len(train_indices))
    print("Validation samples are:", len(validation_indices))
    print("Test samples are:", len(test_labels))


if __name__ == "__main__":
    main()