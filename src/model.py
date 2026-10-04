"""Shared CNN definition (used by train.py and evaluate.py)."""
import torch.nn as nn


def build_model(num_filters, dropout_rate, dense_units, image_size=32):
    f = num_filters
    flat = 2 * f * (image_size // 4) ** 2
    return nn.Sequential(
        nn.Conv2d(3, f, 3, padding=1), nn.BatchNorm2d(f), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(f, 2 * f, 3, padding=1), nn.BatchNorm2d(2 * f), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(flat, dense_units),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(dense_units, 10),    
        ) 