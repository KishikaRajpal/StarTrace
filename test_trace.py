"""
test_trace.py - Quick unit test verifying pipeline ingestion, conditioning, BLS,
dual-branch tensor shapes, and model classification for TIC 25155310.
"""

import sys
import io

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import torch

from pipeline import process_lightcurve, PRESET_METADATA
from model import DualBranchCNN, get_model, classify_lightcurve, CLASSES

def test_pipeline_and_model():
    print("=" * 60)
    print("Testing StarTrace: TIC 25155310 (Kepler-1649c)")
    print("=" * 60)

    # 1. Test Model Architecture Directly
    print("\n[Step 1] Verifying DualBranchCNN Architecture Shapes...")
    model = DualBranchCNN(num_classes=4)
    model.eval()

    dummy_global = torch.randn(2, 1, 2001)
    dummy_local = torch.randn(2, 1, 201)

    with torch.no_grad():
        out = model(dummy_global, dummy_local)
    
    assert out.shape == (2, 4), f"Expected shape (2, 4), got {out.shape}"
    print(f"  [OK] Forward pass successful! Output logits shape: {out.shape}")

    # 2. Test Pipeline Processing
    print("\n[Step 2] Executing End-to-End Pipeline on TIC 25155310...")
    result = process_lightcurve(
        source_type="NASA MAST Live Stream",
        tic_id="25155310",
        window_length=101,
        min_period=0.5,
        max_period=15.0
    )

    time = result["time"]
    raw_flux = result["raw_flux"]
    detrended = result["detrended_flux"]
    bls_stats = result["bls_stats"]
    dual = result["dual_views"]

    print(f"  [OK] Data ingested: {len(time)} cadence points (Fallback: {result['is_fallback']})")
    print(f"  [OK] Detrended flux median: {np.median(detrended):.5f}")
    print(f"  [OK] BLS detected period: {bls_stats['period_days']:.4f} days")
    print(f"  [OK] BLS detected depth: {bls_stats['depth_pct']:.4f}%")
    print(f"  [OK] BLS detected duration: {bls_stats['duration_hours']:.2f} hours")
    print(f"  [OK] BLS SNR: {bls_stats['snr']:.2f}")

    global_view = dual["global_view"]
    local_view = dual["local_view"]

    assert global_view.shape == (2001,), f"Global view shape mismatch: {global_view.shape}"
    assert local_view.shape == (201,), f"Local view shape mismatch: {local_view.shape}"
    assert not np.isnan(global_view).any(), "Global view contains NaNs"
    assert not np.isnan(local_view).any(), "Local view contains NaNs"
    print(f"  [OK] Dual-view binning validated: Global={global_view.shape}, Local={local_view.shape}")

    # 3. Test Trained/Calibrated Inference
    print("\n[Step 3] Testing Classification Inference...")
    calibrated_model = get_model("weights.pth")
    preds = classify_lightcurve(calibrated_model, global_view, local_view, bls_stats=bls_stats)

    print(f"  [OK] Predicted Class: {preds['prediction']}")
    print(f"  [OK] Confidence: {preds['confidence'] * 100:.2f}%")
    print("  [OK] Probability Distribution:")
    for c_name, p_val in preds["probabilities"].items():
        print(f"     - {c_name:25s}: {p_val * 100:6.2f}%")

    assert preds["prediction"] in CLASSES, "Invalid class predicted"
    assert 0.0 <= preds["confidence"] <= 1.0, "Confidence out of range [0, 1]"

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED CLEANLY! Ready for Live Demo.")
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline_and_model()
