"""
api/health.py - FastAPI health check and system status endpoint.
"""

import sys
import os
import platform
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from app.database.session import get_db, get_db_info

router = APIRouter(prefix="/api", tags=["Health"])


@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Returns system health, database status, and model availability."""
    from ml.inference import get_model_status
    model_info = get_model_status()
    db_info = get_db_info()

    return {
        "status": "ok",
        "version": "1.0.0",
        "platform": platform.system(),
        "python_version": sys.version,
        "database": db_info,
        "model": {
            "status": model_info.get("status"),
            "is_trained": model_info.get("is_trained"),
            "device": model_info.get("device"),
            "total_parameters": model_info.get("total_parameters")
        }
    }


@router.get("/samples")
def list_sample_datasets():
    """Returns list of built-in sample datasets for demonstration analysis."""
    return {
        "samples": [
            {
                "tic_id": "25155310",
                "name": "Kepler-1649c (Exoplanet Transit)",
                "category": "Exoplanet Transit",
                "description": "Earth-sized planet in habitable zone of M-dwarf star.",
                "expected_period_days": 8.689,
                "expected_depth_pct": 0.22
            },
            {
                "tic_id": "149603524",
                "name": "Eclipsing Binary System",
                "category": "Eclipsing Binary",
                "description": "Detached eclipsing binary with deep primary and secondary minima.",
                "expected_period_days": 2.342,
                "expected_depth_pct": 8.5
            },
            {
                "tic_id": "38846515",
                "name": "Background Blend Source",
                "category": "Background Blend",
                "description": "Aperture contamination from background eclipsing binary.",
                "expected_period_days": 3.754,
                "expected_depth_pct": 0.85
            },
            {
                "tic_id": "261136638",
                "name": "Active Variable Star",
                "category": "Stellar Noise / Activity",
                "description": "Rotating active star with starspot modulation and flares.",
                "expected_period_days": 1.428,
                "expected_depth_pct": 0.55
            }
        ]
    }
