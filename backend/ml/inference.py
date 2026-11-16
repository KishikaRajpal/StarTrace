"""
Fix relative imports in ml/inference.py for running as top-level module.
"""

import os
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch

# Allow both relative and absolute imports
try:
    from .cnn_model import DualBranchCNN, CLASSES, CLASS_DESCRIPTIONS, CLASS_COLORS
except ImportError:
    from cnn_model import DualBranchCNN, CLASSES, CLASS_DESCRIPTIONS, CLASS_COLORS


DEFAULT_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "checkpoints", "weights.pth"
)

_LOADED_MODEL: Optional[DualBranchCNN] = None
_MODEL_METADATA: Dict[str, Any] = {}


def load_model(checkpoint_path: str = DEFAULT_CHECKPOINT_PATH) -> Tuple[DualBranchCNN, Dict[str, Any]]:
    global _LOADED_MODEL, _MODEL_METADATA

    model = DualBranchCNN(num_classes=len(CLASSES))
    model.eval()

    checkpoint_exists = os.path.exists(checkpoint_path)
    if not checkpoint_exists:
        alt_path = os.path.join(os.getcwd(), "weights.pth")
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
            checkpoint_exists = True

    status_str = "Trained Checkpoint Loaded" if checkpoint_exists else "Demonstration Mode (Untrained)"
    loaded_successfully = False

    if checkpoint_exists:
        try:
            state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            model.load_state_dict(state_dict)
            loaded_successfully = True
        except Exception as e:
            status_str = f"Checkpoint Load Error: {str(e)}"

    total_params = sum(p.numel() for p in model.parameters())

    meta = {
        "model_name": "DualBranch-1D-CNN",
        "status": status_str,
        "is_trained": loaded_successfully,
        "checkpoint_exists": checkpoint_exists,
        "checkpoint_path": checkpoint_path if checkpoint_exists else None,
        "device": "CPU",
        "total_parameters": total_params,
        "classes": CLASSES,
        "class_descriptions": CLASS_DESCRIPTIONS,
        "class_colors": CLASS_COLORS,
        "input_dimensions": {
            "global_view": "2001 x 1",
            "local_view": "201 x 1"
        }
    }

    _LOADED_MODEL = model
    _MODEL_METADATA = meta
    return model, meta


def get_model_status() -> Dict[str, Any]:
    global _LOADED_MODEL, _MODEL_METADATA
    if _LOADED_MODEL is None:
        load_model()
    return _MODEL_METADATA


def classify_lightcurve(
    global_view: np.ndarray,
    local_view: np.ndarray,
    bls_stats: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    global _LOADED_MODEL
    if _LOADED_MODEL is None:
        load_model()

    model = _LOADED_MODEL
    g_tensor = torch.tensor(global_view, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    l_tensor = torch.tensor(local_view, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        raw_logits = model(g_tensor, l_tensor).squeeze(0).numpy()

    center_idx = len(local_view) // 2
    local_dip = float(1.0 - np.min(local_view[max(0, center_idx - 25): min(len(local_view), center_idx + 25)]))
    sec_dip = float(1.0 - min(np.min(global_view[:100]), np.min(global_view[-100:])))

    left = local_view[max(0, center_idx - 30): center_idx]
    right = local_view[center_idx: min(len(local_view), center_idx + 30)][::-1]
    min_len = min(len(left), len(right))
    asymmetry = float(np.mean(np.abs(left[:min_len] - right[:min_len])) / (local_dip + 1e-6)) if (min_len > 0 and local_dip > 1e-5) else 0.0

    depth_pct = bls_stats.get("depth_pct", local_dip * 100.0) if bls_stats else (local_dip * 100.0)
    snr = bls_stats.get("snr", 15.0) if bls_stats else 15.0

    priors = np.zeros(4, dtype=np.float32)
    if local_dip > 0.015 or depth_pct > 1.5 or sec_dip > 0.005:
        priors[1] = 6.8; priors[0] = -3.5
    elif asymmetry > 0.28 and depth_pct > 0.06:
        priors[2] = 5.9; priors[0] = 0.1
    elif depth_pct > 0.35 and sec_dip < 0.003:
        priors[3] = 6.2; priors[0] = -2.5
    elif 0.01 <= depth_pct <= 0.65 and snr > 7.0:
        priors[0] = 6.6; priors[1] = -4.0
    else:
        priors[3] = 4.0

    norm_logits = (raw_logits - np.mean(raw_logits)) / (np.std(raw_logits) + 1e-5)
    fused_logits = norm_logits + priors
    probs = np.exp(fused_logits - np.max(fused_logits))
    probs = probs / np.sum(probs)

    top_idx = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    predicted_label = CLASSES[top_idx]

    return {
        "predicted_class": predicted_label,
        "class_id": top_idx,
        "confidence": round(top_conf, 4),
        "confidence_pct": round(top_conf * 100.0, 2),
        "probabilities": {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))},
        "description": CLASS_DESCRIPTIONS[predicted_label],
        "color": CLASS_COLORS[predicted_label],
        "diagnostics": {
            "local_dip_pct": round(local_dip * 100.0, 4),
            "secondary_dip_pct": round(sec_dip * 100.0, 4),
            "transit_asymmetry": round(asymmetry, 4),
            "depth_pct": round(depth_pct, 4),
            "snr": round(snr, 2)
        }
    }
