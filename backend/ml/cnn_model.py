"""
cnn_model.py - Dual-Branch 1D-CNN PyTorch Architecture for Exoplanet Transit Classification.

Architecture:
  - Global Branch: 2001-point input spanning full orbital phase [-0.5, 0.5]
  - Local Branch: 201-point input zoomed on primary transit window [-0.05, 0.05]
  - Fusion Head: Concatenates extracted macro & micro features, passes through
    dense layers with dropout, and outputs 4 astrophysical classification probabilities:
      0: Exoplanet Transit
      1: Eclipsing Binary
      2: Background Blend
      3: Stellar Noise / Activity
"""

import os
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = [
    "Exoplanet Transit",
    "Eclipsing Binary",
    "Background Blend",
    "Stellar Noise / Activity"
]

CLASS_DESCRIPTIONS = {
    "Exoplanet Transit": "Periodic shallow U-shaped occultation caused by an exoplanet traversing the host star.",
    "Eclipsing Binary": "Deep V-shaped primary transit exhibiting secondary eclipses or ellipsoidal variation.",
    "Background Blend": "Aperture contamination from an unresolved background eclipsing binary source.",
    "Stellar Noise / Activity": "Non-transit stellar pulsations, rotational starspots, flares, or instrumental noise."
}

CLASS_COLORS = {
    "Exoplanet Transit": "#00f2fe",        # Electric Cyan
    "Eclipsing Binary": "#ff007f",         # Hot Pink / Magenta
    "Background Blend": "#ffd166",         # Cosmic Amber
    "Stellar Noise / Activity": "#06d6a0"  # Emerald
}


class DualBranchCNN(nn.Module):
    """
    Dual-Branch 1D Convolutional Neural Network for Astronomical Transit Photometry.
    Processes both macro-orbital (global) and micro-ingress/egress (local) representations.
    """
    def __init__(self, num_classes: int = 4):
        super(DualBranchCNN, self).__init__()

        # Global Branch: (Batch, 1, 2001) -> Conv1D -> MaxPool -> (Batch, 16000)
        self.global_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 2001 -> 1000

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 1000 -> 500

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 500 -> 250
            nn.Flatten()                            # 64 * 250 = 16000
        )

        # Local Branch: (Batch, 1, 201) -> Conv1D -> MaxPool -> (Batch, 1600)
        self.local_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 201 -> 100

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 100 -> 50

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 50 -> 25
            nn.Flatten()                            # 64 * 25 = 1600
        )

        # Fusion Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(16000 + 1600, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_global: torch.Tensor, x_local: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for dual inputs.
        Accepts (Batch, 2001) or (Batch, 1, 2001) for global,
        and (Batch, 201) or (Batch, 1, 201) for local.
        """
        if x_global.dim() == 2:
            x_global = x_global.unsqueeze(1)
        if x_local.dim() == 2:
            x_local = x_local.unsqueeze(1)

        f_global = self.global_branch(x_global)
        f_local = self.local_branch(x_local)

        fused = torch.cat([f_global, f_local], dim=1)
        logits = self.classifier(fused)
        return logits
