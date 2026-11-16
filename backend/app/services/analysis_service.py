"""
app/services/analysis_service.py - Core orchestration service using flat sys.path imports.
"""

import os
import sys
import uuid
import logging
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any

# Insert backend root so astronomy/ml can be imported as top-level
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

logger = logging.getLogger("startrace.service")

VIZ_MAX_RAW_PTS = 3000


def run_analysis_pipeline(
    tic_id: str,
    db,
    min_period: float = 0.5,
    max_period: float = 15.0,
    csv_content: Optional[bytes] = None,
    csv_filename: Optional[str] = None
) -> Dict[str, Any]:
    """Full end-to-end StarTrace analysis pipeline."""
    from astronomy.data_loader import fetch_tess_lightcurve, load_csv_lightcurve
    from astronomy.preprocessing import condition_and_detrend_lightcurve
    from astronomy.period_detection import run_bls_detection
    from astronomy.phase_folding import compute_phase_and_dual_views
    from astronomy.parameter_estimation import estimate_transit_parameters
    from ml.inference import classify_lightcurve, get_model_status, load_model
    from app.database.models import Analysis, ClassificationResult
    from app.core.config import settings

    analysis_id = str(uuid.uuid4())

    # Step 1: Data ingestion
    try:
        if csv_content is not None:
            time, flux, ingest_meta = load_csv_lightcurve(csv_content, csv_filename or "data.csv")
        else:
            time, flux, ingest_meta = fetch_tess_lightcurve(str(tic_id).strip(), timeout_sec=8)
    except Exception as e:
        return _make_error_result(analysis_id, tic_id, str(e))

    is_synthetic = bool(ingest_meta.get("is_synthetic", True))
    data_source = str(ingest_meta.get("source", "Unknown"))

    # Step 2: Preprocessing
    try:
        t_clean, f_clean, trend, detrended, preproc_summary = condition_and_detrend_lightcurve(
            time, flux, window_length=101, polyorder=2, sigma=5.0
        )
    except Exception as e:
        return _make_error_result(analysis_id, tic_id, f"Preprocessing failed: {e}", data_source, is_synthetic)

    # Step 3: BLS period detection
    try:
        bls_stats, periodogram_data = run_bls_detection(
            t_clean, detrended, min_period=min_period, max_period=max_period
        )
    except Exception as e:
        return _make_error_result(analysis_id, tic_id, f"BLS detection failed: {e}", data_source, is_synthetic)

    # Step 4: Phase folding and dual views
    try:
        views = compute_phase_and_dual_views(
            t_clean, detrended,
            period=bls_stats["period_days"],
            t0=bls_stats["transit_epoch_bjd"]
        )
    except Exception as e:
        return _make_error_result(analysis_id, tic_id, f"Phase folding failed: {e}", data_source, is_synthetic)

    # Step 5: AI classification
    try:
        load_model(settings.CHECKPOINT_PATH)
        clf_result = classify_lightcurve(views["global_view"], views["local_view"], bls_stats=bls_stats)
        model_status = get_model_status()
        model_trained = bool(model_status.get("is_trained", False))
    except Exception as e:
        logger.error(f"Classification error: {e}")
        clf_result = {
            "predicted_class": "Stellar Noise / Activity",
            "class_id": 3,
            "confidence": 0.5,
            "confidence_pct": 50.0,
            "probabilities": {"Exoplanet Transit": 0.25, "Eclipsing Binary": 0.25,
                              "Background Blend": 0.25, "Stellar Noise / Activity": 0.25},
            "description": "Classification unavailable.",
            "color": "#888888",
            "diagnostics": {}
        }
        model_trained = False

    # Step 6: Parameter estimation
    try:
        transit_params = estimate_transit_parameters(
            bls_stats, clf_result["predicted_class"], clf_result["confidence"]
        )
    except Exception as e:
        logger.error(f"Parameter estimation error: {e}")
        transit_params = {}

    # Step 7: Persist to database
    probs = clf_result.get("probabilities", {})
    db_analysis = Analysis(
        id=analysis_id,
        tic_id=str(tic_id).strip(),
        data_source=data_source,
        is_synthetic=is_synthetic,
        status="completed",
        predicted_class=clf_result["predicted_class"],
        class_id=clf_result.get("class_id"),
        confidence_score=clf_result["confidence"],
        orbital_period_days=bls_stats.get("period_days"),
        transit_depth_pct=bls_stats.get("depth_pct"),
        transit_depth_ppm=bls_stats.get("depth_ppm"),
        transit_duration_hours=bls_stats.get("duration_hours"),
        transit_epoch_bjd=bls_stats.get("transit_epoch_bjd"),
        snr=bls_stats.get("snr"),
        max_bls_power=bls_stats.get("max_power"),
        num_transits=bls_stats.get("num_transits_observed"),
        planet_radius_earth=transit_params.get("estimated_planet_radius_earth"),
        semi_major_axis_au=transit_params.get("estimated_semi_major_axis_au"),
        equilibrium_temp_k=transit_params.get("estimated_equilibrium_temp_k"),
        completed_at=datetime.utcnow()
    )
    db_clf = ClassificationResult(
        analysis_id=analysis_id,
        prob_exoplanet_transit=probs.get("Exoplanet Transit", 0.0),
        prob_eclipsing_binary=probs.get("Eclipsing Binary", 0.0),
        prob_stellar_blend=probs.get("Background Blend", 0.0),
        prob_stellar_noise=probs.get("Stellar Noise / Activity", 0.0)
    )

    try:
        db.add(db_analysis)
        db.add(db_clf)
        db.commit()
        db.refresh(db_analysis)
    except Exception as db_err:
        logger.error(f"Database error: {db_err}")
        db.rollback()

    _save_analysis_data(analysis_id, t_clean, f_clean, trend, detrended, settings.STORAGE_DIR)

    viz = _build_visualization(t_clean, f_clean, trend, detrended, bls_stats, periodogram_data, views)

    return {
        "analysis_id": analysis_id,
        "tic_id": str(tic_id).strip(),
        "data_source": data_source,
        "is_synthetic": is_synthetic,
        "status": "completed",
        "model_trained": model_trained,
        "classification": clf_result,
        "transit_parameters": transit_params,
        "bls_stats": bls_stats,
        "preprocessing_summary": preproc_summary,
        "visualization": viz,
        "created_at": db_analysis.created_at.isoformat() if db_analysis.created_at else None,
        "completed_at": db_analysis.completed_at.isoformat() if db_analysis.completed_at else None
    }


def _make_error_result(analysis_id, tic_id, error_msg, data_source="Unknown", is_synthetic=True):
    return {
        "analysis_id": analysis_id, "tic_id": str(tic_id), "data_source": data_source,
        "is_synthetic": is_synthetic, "status": "failed", "model_trained": False,
        "error": error_msg, "classification": None, "transit_parameters": None,
        "bls_stats": None, "preprocessing_summary": None, "visualization": None
    }


def _save_analysis_data(analysis_id, time, flux, trend, detrended, storage_dir):
    try:
        os.makedirs(storage_dir, exist_ok=True)
        path = os.path.join(storage_dir, f"{analysis_id}.npz")
        np.savez_compressed(path, time=time, flux=flux, trend=trend, detrended=detrended)
        return path
    except Exception as e:
        logger.warning(f"Could not save data: {e}")
        return None


def _build_visualization(t_clean, f_clean, trend, detrended, bls_stats, periodogram_data, views):
    n = len(t_clean)
    step = max(1, n // VIZ_MAX_RAW_PTS)
    return {
        "raw_time": t_clean[::step].tolist(),
        "raw_flux": f_clean[::step].tolist(),
        "clean_time": t_clean[::step].tolist(),
        "detrended_flux": detrended[::step].tolist(),
        "trend": trend[::step].tolist(),
        "bls_periods": periodogram_data.get("periods", []),
        "bls_powers": periodogram_data.get("powers", []),
        "bls_best_period": bls_stats.get("period_days", 0.0),
        "bls_max_power": bls_stats.get("max_power", 0.0),
        "folded_phase": views.get("folded_phase_scatter", []),
        "folded_flux": views.get("folded_flux_scatter", []),
        "global_phase": views["centers_global"].tolist(),
        "global_flux": views["global_view"].tolist(),
        "local_phase": views["centers_local"].tolist(),
        "local_flux": views["local_view"].tolist()
    }


def get_analysis_by_id(analysis_id: str, db) -> Optional[Dict[str, Any]]:
    from app.database.models import Analysis, ClassificationResult
    from app.core.config import settings
    from ml.cnn_model import CLASS_DESCRIPTIONS, CLASS_COLORS

    record = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not record:
        return None
    clf_rec = db.query(ClassificationResult).filter(ClassificationResult.analysis_id == analysis_id).first()

    probs = {}
    if clf_rec:
        probs = {
            "Exoplanet Transit": clf_rec.prob_exoplanet_transit or 0.0,
            "Eclipsing Binary": clf_rec.prob_eclipsing_binary or 0.0,
            "Background Blend": clf_rec.prob_stellar_blend or 0.0,
            "Stellar Noise / Activity": clf_rec.prob_stellar_noise or 0.0
        }

    predicted = record.predicted_class or "Unknown"
    bls = {
        "period_days": record.orbital_period_days,
        "transit_epoch_bjd": record.transit_epoch_bjd,
        "duration_hours": record.transit_duration_hours,
        "depth_pct": record.transit_depth_pct,
        "depth_ppm": record.transit_depth_ppm,
        "snr": record.snr,
        "max_power": record.max_bls_power,
        "num_transits_observed": record.num_transits
    }

    result = {
        "analysis_id": record.id,
        "tic_id": record.tic_id,
        "data_source": record.data_source,
        "is_synthetic": record.is_synthetic,
        "status": record.status,
        "model_trained": True,
        "classification": {
            "predicted_class": predicted,
            "class_id": record.class_id or 3,
            "confidence": record.confidence_score or 0.0,
            "confidence_pct": round((record.confidence_score or 0.0) * 100, 2),
            "probabilities": probs,
            "description": CLASS_DESCRIPTIONS.get(predicted, ""),
            "color": CLASS_COLORS.get(predicted, "#888")
        },
        "bls_stats": bls,
        "transit_parameters": {
            "is_candidate": predicted == "Exoplanet Transit",
            "status_label": predicted,
            "orbital_period_days": record.orbital_period_days or 0.0,
            "orbital_period_err": 0.0,
            "transit_epoch_bjd": record.transit_epoch_bjd or 0.0,
            "transit_depth_pct": record.transit_depth_pct or 0.0,
            "transit_depth_ppm": record.transit_depth_ppm or 0.0,
            "transit_duration_hours": record.transit_duration_hours or 0.0,
            "snr": record.snr or 0.0,
            "estimated_planet_radius_earth": record.planet_radius_earth,
            "estimated_semi_major_axis_au": record.semi_major_axis_au,
            "estimated_equilibrium_temp_k": record.equilibrium_temp_k,
            "interpretation": ""
        },
        "visualization": None,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "completed_at": record.completed_at.isoformat() if record.completed_at else None
    }

    # Try loading visualization from stored npz
    npz_path = os.path.join(settings.STORAGE_DIR, f"{analysis_id}.npz")
    if os.path.exists(npz_path):
        try:
            from astronomy.phase_folding import compute_phase_and_dual_views
            loaded = np.load(npz_path)
            period = record.orbital_period_days or 5.0
            epoch = record.transit_epoch_bjd or 0.0
            t = loaded["time"]; detrended = loaded["detrended"]
            f_clean = loaded["flux"]; trend = loaded["trend"]
            n = len(t); step = max(1, n // VIZ_MAX_RAW_PTS)
            views = compute_phase_and_dual_views(t, detrended, period=period, t0=epoch)
            result["visualization"] = {
                "raw_time": t[::step].tolist(), "raw_flux": f_clean[::step].tolist(),
                "clean_time": t[::step].tolist(), "detrended_flux": detrended[::step].tolist(),
                "trend": trend[::step].tolist(),
                "bls_periods": [], "bls_powers": [], "bls_best_period": period, "bls_max_power": 0.0,
                "folded_phase": views.get("folded_phase_scatter", []),
                "folded_flux": views.get("folded_flux_scatter", []),
                "global_phase": views["centers_global"].tolist(),
                "global_flux": views["global_view"].tolist(),
                "local_phase": views["centers_local"].tolist(),
                "local_flux": views["local_view"].tolist()
            }
        except Exception:
            pass

    return result


def list_analyses(db, skip=0, limit=50, filter_class=None, search_tic=None):
    from app.database.models import Analysis
    query = db.query(Analysis)
    if filter_class:
        query = query.filter(Analysis.predicted_class == filter_class)
    if search_tic:
        query = query.filter(Analysis.tic_id.ilike(f"%{search_tic}%"))
    total = query.count()
    records = query.order_by(Analysis.created_at.desc()).offset(skip).limit(limit).all()
    return {"total": total, "items": [_to_summary(r) for r in records]}


def delete_analysis(analysis_id: str, db) -> bool:
    from app.database.models import Analysis
    from app.core.config import settings
    record = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not record:
        return False
    npz_path = os.path.join(settings.STORAGE_DIR, f"{analysis_id}.npz")
    if os.path.exists(npz_path):
        try:
            os.remove(npz_path)
        except Exception:
            pass
    db.delete(record)
    db.commit()
    return True


def get_dashboard_stats(db):
    from app.database.models import Analysis
    total = db.query(Analysis).count()
    exo = db.query(Analysis).filter(Analysis.predicted_class == "Exoplanet Transit").count()
    eb = db.query(Analysis).filter(Analysis.predicted_class == "Eclipsing Binary").count()
    blend = db.query(Analysis).filter(Analysis.predicted_class == "Background Blend").count()
    noise = db.query(Analysis).filter(Analysis.predicted_class == "Stellar Noise / Activity").count()
    recent = db.query(Analysis).order_by(Analysis.created_at.desc()).limit(10).all()
    return {
        "total_analyses": total, "exoplanet_candidates": exo,
        "eclipsing_binaries": eb, "background_blends": blend,
        "stellar_noise": noise, "recent_analyses": [_to_summary(r) for r in recent]
    }


def _to_summary(record):
    return {
        "analysis_id": record.id, "tic_id": record.tic_id, "status": record.status,
        "data_source": record.data_source, "is_synthetic": record.is_synthetic,
        "predicted_class": record.predicted_class, "confidence": record.confidence_score,
        "orbital_period_days": record.orbital_period_days, "snr": record.snr,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "completed_at": record.completed_at.isoformat() if record.completed_at else None
    }
