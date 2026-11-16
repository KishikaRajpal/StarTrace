"""
api/analysis.py - FastAPI router for analysis endpoints.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File, Form, Query
from sqlalchemy.orm import Session

from ..database.session import get_db
from ..services.analysis_service import (
    run_analysis_pipeline,
    get_analysis_by_id,
    list_analyses,
    delete_analysis,
    get_dashboard_stats
)

logger = logging.getLogger("startrace.api")
router = APIRouter(prefix="/api/analysis", tags=["Analysis"])

# In-memory job status store for background tasks
_job_status: dict = {}


@router.post("/run")
def run_analysis(
    background_tasks: BackgroundTasks,
    tic_id: str = Form(...),
    min_period: float = Form(0.5),
    max_period: float = Form(15.0),
    db: Session = Depends(get_db)
):
    """
    Accepts a TIC ID and runs the full StarTrace analysis pipeline.
    Returns complete results synchronously (suitable for typical student laptop performance).
    """
    tic_clean = tic_id.strip().replace("TIC", "").replace("tic", "").strip()
    if not tic_clean or not tic_clean.replace("sim_", "").replace("_", "").replace("-", "").isdigit():
        # Accept numeric and "sim_" prefixed IDs
        if not tic_clean:
            raise HTTPException(status_code=422, detail="TIC ID must not be empty.")

    try:
        result = run_analysis_pipeline(
            tic_id=tic_clean,
            db=db,
            min_period=float(min_period),
            max_period=float(max_period)
        )
        return result
    except Exception as e:
        logger.error(f"Pipeline error for TIC {tic_clean}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/upload")
def run_analysis_from_csv(
    file: UploadFile = File(...),
    min_period: float = Form(0.5),
    max_period: float = Form(15.0),
    db: Session = Depends(get_db)
):
    """
    Accepts a user-uploaded CSV light curve file and runs analysis.
    Expected CSV columns: time (or bjd / btjd) and flux (or pdcsap_flux / sap_flux).
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Only CSV files are supported.")
    if file.size and file.size > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="CSV file must be under 20 MB.")

    try:
        content = file.file.read()
        result = run_analysis_pipeline(
            tic_id="Uploaded-CSV",
            db=db,
            min_period=float(min_period),
            max_period=float(max_period),
            csv_content=content,
            csv_filename=file.filename
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CSV upload analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"CSV analysis failed: {str(e)}")


@router.get("/dashboard")
def get_dashboard(db: Session = Depends(get_db)):
    """Returns aggregated dashboard statistics from the database."""
    return get_dashboard_stats(db)


@router.get("")
def list_analysis_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    filter_class: Optional[str] = Query(None),
    search_tic: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Returns paginated analysis history with optional filtering."""
    return list_analyses(db, skip=skip, limit=limit, filter_class=filter_class, search_tic=search_tic)


@router.get("/{analysis_id}")
def get_analysis(analysis_id: str, db: Session = Depends(get_db)):
    """Returns full analysis result for a given analysis ID."""
    result = get_analysis_by_id(analysis_id, db)
    if not result:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found.")
    return result


@router.delete("/{analysis_id}")
def delete_analysis_record(analysis_id: str, db: Session = Depends(get_db)):
    """Deletes an analysis record and its stored data files."""
    success = delete_analysis(analysis_id, db)
    if not success:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found.")
    return {"deleted": True, "analysis_id": analysis_id}


@router.get("/{analysis_id}/visualization")
def get_visualization(analysis_id: str, db: Session = Depends(get_db)):
    """Returns visualization-ready data for a previous analysis."""
    result = get_analysis_by_id(analysis_id, db)
    if not result:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found.")
    return result.get("visualization") or {}
