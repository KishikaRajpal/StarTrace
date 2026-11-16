"""
dataset.py - Dataset generation and PyTorch Dataset module for training
and evaluating the StarTrace Dual-Branch 1D-CNN.
"""

from typing import Tuple, List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset

from ..astronomy.data_loader import generate_synthetic_lightcurve, SAMPLE_PRESETS
from ..astronomy.preprocessing import condition_and_detrend_lightcurve
from ..astronomy.period_detection import run_bls_detection
from ..astronomy.phase_folding import compute_phase_and_dual_views
from .cnn_model import CLASSES


class ExoplanetTransitDataset(Dataset):
    """
    PyTorch Dataset serving paired Global (2001) and Local (201) light curve vectors.
    """
    def __init__(self, global_tensors: torch.Tensor, local_tensors: torch.Tensor, labels: torch.Tensor):
        self.global_tensors = global_tensors
        self.local_tensors = local_tensors
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.global_tensors[idx], self.local_tensors[idx], self.labels[idx]


def generate_synthetic_training_data(
    samples_per_class: int = 100,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesizes and processes light curves across all four astrophysical classes.
    Extracts 2001-point global and 201-point local vectors.
    """
    rng = np.random.RandomState(seed)

    global_list: List[np.ndarray] = []
    local_list: List[np.ndarray] = []
    label_list: List[int] = []

    for class_id, class_name in enumerate(CLASSES):
        for i in range(samples_per_class):
            fake_tic = f"sim_{class_id}_{i}_{rng.randint(100000, 999999)}"
            # Generate light curve
            t, f, _ = generate_synthetic_lightcurve(
                tic_id=fake_tic,
                category_hint=class_name,
                cadence_min=2.0,
                duration_days=20.0
            )

            # Conditioning
            try:
                t_clean, f_clean, trend, detrended, _ = condition_and_detrend_lightcurve(t, f)
                bls_stats, _ = run_bls_detection(t_clean, detrended, min_period=0.5, max_period=10.0)
                views = compute_phase_and_dual_views(
                    t_clean,
                    detrended,
                    period=bls_stats["period_days"],
                    t0=bls_stats["transit_epoch_bjd"]
                )

                global_list.append(views["global_view"])
                local_list.append(views["local_view"])
                label_list.append(class_id)
            except Exception:
                # If fitting fails on noisy sample, skip
                continue

    X_global = np.array(global_list, dtype=np.float32)
    X_local = np.array(local_list, dtype=np.float32)
    y = np.array(label_list, dtype=np.int64)

    return X_global, X_local, y
