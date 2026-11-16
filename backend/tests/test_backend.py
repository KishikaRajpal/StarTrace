"""
tests/test_backend.py - Comprehensive automated tests for StarTrace:
- Astronomy preprocessing & 5-sigma outlier removal
- Astropy BLS period detection
- Phase folding & Dual-View (2001 & 201) binning
- PyTorch 1D-CNN tensor shapes & inference
- Astrophysical parameter estimation
- Database ORM operations
- FastAPI HTTP REST endpoints
"""

import os
import sys
import pytest
import numpy as np
from fastapi.testclient import TestClient

# Ensure backend root is on Python path
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from astronomy.data_loader import generate_synthetic_lightcurve, load_csv_lightcurve, SAMPLE_PRESETS
from astronomy.preprocessing import condition_and_detrend_lightcurve
from astronomy.period_detection import run_bls_detection
from astronomy.phase_folding import compute_phase_and_dual_views
from astronomy.parameter_estimation import estimate_transit_parameters
from ml.cnn_model import DualBranchCNN, CLASSES
from ml.inference import load_model, classify_lightcurve, get_model_status
from app.database.session import init_db, get_db
from main import app


@pytest.fixture(scope="session")
def client():
    """Initialises test database and yields FastAPI TestClient."""
    init_db()
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------
# 1. Preprocessing & Outlier Removal Tests
# ---------------------------------------------------------
def test_preprocessing_and_outlier_removal():
    time, flux, meta = generate_synthetic_lightcurve("25155310", duration_days=10.0, cadence_min=5.0)

    # Inject artificial extreme positive cosmic ray spikes and NaNs
    flux_with_spikes = np.copy(flux)
    flux_with_spikes[10] = 5.0  # Massive spike
    flux_with_spikes[20] = np.nan
    flux_with_spikes[30] = np.inf

    t_clean, f_clean, trend, detrended, summary = condition_and_detrend_lightcurve(
        time, flux_with_spikes, window_length=101, polyorder=2, sigma=5.0
    )

    assert len(t_clean) == len(detrended)
    assert not np.any(np.isnan(detrended))
    assert not np.any(np.isinf(detrended))
    assert np.max(f_clean) < 3.0  # Confirms cosmic ray spike was clipped
    assert summary["outliers_removed"] >= 1
    # Baseline median should be normalized near 1.0
    assert np.isclose(np.median(detrended), 1.0, atol=0.01)


# ---------------------------------------------------------
# 2. Box Least Squares (BLS) Detection Tests
# ---------------------------------------------------------
def test_bls_period_detection():
    # Use preset with known period = 8.689 days
    t, f, meta = generate_synthetic_lightcurve("25155310", duration_days=25.0, cadence_min=3.0)
    t_clean, f_clean, trend, detrended, _ = condition_and_detrend_lightcurve(t, f)

    bls_stats, periodogram = run_bls_detection(
        t_clean, detrended, min_period=4.0, max_period=12.0
    )

    detected_period = bls_stats["period_days"]
    expected_period = meta["expected_period"]

    # Detected period must be within 3% of true period
    assert np.isclose(detected_period, expected_period, rtol=0.03)
    assert bls_stats["depth_pct"] > 0.05
    assert bls_stats["snr"] > 10.0
    assert len(periodogram["periods"]) > 50


# ---------------------------------------------------------
# 3. Phase Folding & Input Dimensions Tests
# ---------------------------------------------------------
def test_phase_folding_dimensions():
    t, f, _ = generate_synthetic_lightcurve("25155310", duration_days=15.0)
    t_clean, _, _, detrended, _ = condition_and_detrend_lightcurve(t, f)

    views = compute_phase_and_dual_views(
        t_clean, detrended, period=8.689, t0=1.2, global_bins=2001, local_bins=201
    )

    # Check exact model input dimensions: 2001 and 201
    assert len(views["global_view"]) == 2001
    assert len(views["local_view"]) == 201
    assert not np.any(np.isnan(views["global_view"]))
    assert not np.any(np.isnan(views["local_view"]))


# ---------------------------------------------------------
# 4. PyTorch Model Forward Pass & Inference Tests
# ---------------------------------------------------------
def test_cnn_model_forward_and_inference():
    import torch
    model = DualBranchCNN(num_classes=4)
    model.eval()

    # Create dummy batch (Batch=2, Channel=1, Length=2001 / 201)
    x_global = torch.randn(2, 1, 2001)
    x_local = torch.randn(2, 1, 201)

    with torch.no_grad():
        logits = model(x_global, x_local)

    assert logits.shape == (2, 4)

    # Test full inference helper
    g_arr = np.ones(2001, dtype=np.float32)
    l_arr = np.ones(201, dtype=np.float32)
    # Simulate a transit dip in local view
    l_arr[90:110] = 0.998

    res = classify_lightcurve(g_arr, l_arr)
    assert "predicted_class" in res
    assert res["predicted_class"] in CLASSES
    assert "probabilities" in res
    assert np.isclose(sum(res["probabilities"].values()), 1.0, atol=0.01)


# ---------------------------------------------------------
# 5. Parameter Estimation Tests
# ---------------------------------------------------------
def test_parameter_estimation():
    dummy_bls = {
        "period_days": 8.689,
        "transit_epoch_bjd": 120.45,
        "duration_hours": 3.2,
        "raw_depth": 0.0022,
        "depth_pct": 0.22,
        "depth_ppm": 2200.0,
        "snr": 35.0
    }

    # Case 1: Exoplanet Transit Candidate
    params_exo = estimate_transit_parameters(dummy_bls, "Exoplanet Transit", 0.98)
    assert params_exo["is_candidate"] is True
    assert params_exo["estimated_planet_radius_earth"] > 0.5
    assert params_exo["estimated_semi_major_axis_au"] > 0.01

    # Case 2: Eclipsing Binary (Planetary estimates should be None)
    params_eb = estimate_transit_parameters(dummy_bls, "Eclipsing Binary", 0.95)
    assert params_eb["is_candidate"] is False
    assert params_eb["estimated_planet_radius_earth"] is None


# ---------------------------------------------------------
# 6. CSV Parsing Tests
# ---------------------------------------------------------
def test_csv_ingestion():
    csv_str = "time,flux\n0.0,1.001\n0.01,0.999\n0.02,1.000\n" + "\n".join([f"{0.03 + i*0.01},{1.0 + (i%5)*0.001}" for i in range(120)])
    time, flux, meta = load_csv_lightcurve(csv_str.encode("utf-8"), "test.csv")
    assert len(time) > 100
    assert meta["source"] == "Uploaded CSV (test.csv)"


# ---------------------------------------------------------
# 7. FastAPI API Endpoints Tests
# ---------------------------------------------------------
def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "database" in data
    assert "model" in data


def test_samples_endpoint(client):
    res = client.get("/api/samples")
    assert res.status_code == 200
    data = res.json()
    assert "samples" in data
    assert len(data["samples"]) == 4


def test_model_endpoints(client):
    res_status = client.get("/api/model/status")
    assert res_status.status_code == 200

    res_metrics = client.get("/api/model/metrics")
    assert res_metrics.status_code == 200
    assert "accuracy" in res_metrics.json()
    assert "confusion_matrix" in res_metrics.json()

    res_arch = client.get("/api/model/architecture")
    assert res_arch.status_code == 200
    assert "total_parameters" in res_arch.json()


def test_full_analysis_run_and_retrieve(client):
    # Run analysis for Kepler-1649c preset
    res = client.post("/api/analysis/run", data={"tic_id": "25155310", "min_period": 0.5, "max_period": 12.0})
    assert res.status_code == 200
    result = res.json()
    assert "analysis_id" in result
    analysis_id = result["analysis_id"]
    assert result["tic_id"] == "25155310"
    assert "classification" in result
    assert "transit_parameters" in result
    assert "visualization" in result

    # Retrieve by ID
    res_get = client.get(f"/api/analysis/{analysis_id}")
    assert res_get.status_code == 200
    assert res_get.json()["analysis_id"] == analysis_id

    # Retrieve history
    res_list = client.get("/api/analysis")
    assert res_list.status_code == 200
    assert res_list.json()["total"] >= 1

    # Check dashboard stats
    res_dash = client.get("/api/analysis/dashboard")
    assert res_dash.status_code == 200
    assert res_dash.json()["total_analyses"] >= 1

    # Test CSV report download
    res_csv = client.get(f"/api/report/{analysis_id}/csv")
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.headers["content-type"]

    # Test PDF report download
    res_pdf = client.get(f"/api/report/{analysis_id}/pdf")
    assert res_pdf.status_code == 200
    assert "application/pdf" in res_pdf.headers["content-type"]
