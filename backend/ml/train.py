"""
train.py - Training pipeline for StarTrace Dual-Branch 1D-CNN.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .cnn_model import DualBranchCNN, CLASSES
from .dataset import generate_synthetic_training_data, ExoplanetTransitDataset


def train_model(
    output_checkpoint: str = "checkpoints/weights.pth",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    samples_per_class: int = 100
) -> DualBranchCNN:
    """
    Trains DualBranchCNN on astronomical light curve views and saves checkpoint.
    """
    print(f"Generating synthetic training dataset ({samples_per_class} per class)...")
    X_global, X_local, y = generate_synthetic_training_data(samples_per_class=samples_per_class)

    dataset = ExoplanetTransitDataset(
        torch.tensor(X_global, dtype=torch.float32),
        torch.tensor(X_local, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )

    n_val = max(8, int(len(dataset) * 0.15))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DualBranchCNN(num_classes=len(CLASSES))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    print(f"Starting training for {epochs} epochs on {n_train} training samples...")

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for g_batch, l_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(g_batch, l_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

        train_acc = correct / max(1, total)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for g_batch, l_batch, y_batch in val_loader:
                logits = model(g_batch, l_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(y_batch)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += len(y_batch)

        val_acc = val_correct / max(1, val_total)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}% | Val Loss: {val_loss/val_total:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(output_checkpoint)), exist_ok=True)
    torch.save(model.state_dict(), output_checkpoint)
    print(f"Model saved successfully to {output_checkpoint}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train StarTrace Dual-Branch 1D-CNN")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/weights.pth")
    args = parser.parse_args()

    train_model(output_checkpoint=args.checkpoint, epochs=args.epochs)
