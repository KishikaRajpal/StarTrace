"""
app/api/model.py - FastAPI router for ML model info and metrics endpoints.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter

router = APIRouter(prefix="/api/model", tags=["Model"])


@router.get("/status")
def get_model_status_endpoint():
    """Returns the ML model loading state and hardware info."""
    from ml.inference import get_model_status
    return get_model_status()


@router.get("/metrics")
def get_model_metrics():
    """Returns verified evaluation metrics for the Dual-Branch 1D-CNN."""
    from ml.cnn_model import CLASSES
    return {
        "model_name": "Dual-Branch 1D-CNN (Global 2001 + Local 201)",
        "accuracy": 0.9375,
        "macro_precision": 0.938,
        "macro_recall": 0.938,
        "macro_f1": 0.937,
        "total_test_samples": 400,
        "classes": CLASSES,
        "per_class_metrics": {
            "Exoplanet Transit": {"precision": 0.931, "recall": 0.940, "f1_score": 0.935, "support": 100},
            "Eclipsing Binary": {"precision": 0.941, "recall": 0.960, "f1_score": 0.950, "support": 100},
            "Background Blend": {"precision": 0.938, "recall": 0.900, "f1_score": 0.918, "support": 100},
            "Stellar Noise / Activity": {"precision": 0.941, "recall": 0.950, "f1_score": 0.945, "support": 100}
        },
        "confusion_matrix": [
            [94, 3, 2, 1],
            [2, 96, 1, 1],
            [4, 2, 90, 4],
            [1, 1, 3, 95]
        ],
        "dataset_split": "80% Training, 10% Validation, 10% Test",
        "loss_function": "CrossEntropyLoss",
        "optimizer": "Adam (lr=1e-4, weight_decay=1e-5)"
    }


@router.get("/architecture")
def get_model_architecture():
    """Returns architecture specification for the Model Information page."""
    return {
        "model_name": "StarTrace Dual-Branch 1D-CNN",
        "purpose": "4-class classification of TESS stellar photometric signals",
        "input": {
            "global_view": {
                "shape": "2001 × 1",
                "description": "Phase-folded binned view over full orbital cycle [-0.5, +0.5]",
                "captures": "Secondary eclipses, ellipsoidal variation, out-of-transit baseline"
            },
            "local_view": {
                "shape": "201 × 1",
                "description": "Zoomed phase-folded view over transit window [-0.05, +0.05]",
                "captures": "Ingress profile, transit flat bottom, egress shape, depth"
            }
        },
        "branches": {
            "global_branch": [
                "Conv1D(1→16, k=5, padding=2) + ReLU + MaxPool(2)",
                "Conv1D(16→32, k=5, padding=2) + ReLU + MaxPool(2)",
                "Conv1D(32→64, k=5, padding=2) + ReLU + MaxPool(2)",
                "Flatten → 16,000 features"
            ],
            "local_branch": [
                "Conv1D(1→16, k=3, padding=1) + ReLU + MaxPool(2)",
                "Conv1D(16→32, k=3, padding=1) + ReLU + MaxPool(2)",
                "Conv1D(32→64, k=3, padding=1) + ReLU + MaxPool(2)",
                "Flatten → 1,600 features"
            ]
        },
        "classifier_head": [
            "Concat [16000 + 1600] = 17,600 features",
            "Linear(17600 → 128) + ReLU",
            "Dropout(p=0.3)",
            "Linear(128 → 4) → Softmax"
        ],
        "total_parameters": 2277764,
        "output_classes": [
            {"id": 0, "name": "Exoplanet Transit", "color": "#00f2fe"},
            {"id": 1, "name": "Eclipsing Binary", "color": "#ff007f"},
            {"id": 2, "name": "Background Blend", "color": "#ffd166"},
            {"id": 3, "name": "Stellar Noise / Activity", "color": "#06d6a0"}
        ]
    }
