"""
schemas/analysis.py - Pydantic request and response schemas for analysis endpoints.
"""

from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
from datetime import datetime


class AnalysisRunRequest(BaseModel):
    tic_id: str = Field(..., min_length=1, max_length=64, description="TESS Input Catalog ID")
    min_period: float = Field(0.5, ge=0.1, le=30.0, description="BLS minimum period (days)")
    max_period: float = Field(15.0, ge=1.0, le=100.0, description="BLS maximum period (days)")

    class Config:
        json_schema_extra = {
            "example": {"tic_id": "25155310", "min_period": 0.5, "max_period": 15.0}
        }


class ProbabilityBreakdown(BaseModel):
    exoplanet_transit: float
    eclipsing_binary: float
    background_blend: float
    stellar_noise: float


class ClassificationResponse(BaseModel):
    predicted_class: str
    class_id: int
    confidence: float
    confidence_pct: float
    probabilities: Dict[str, float]
    description: str
    color: str


class TransitParameters(BaseModel):
    is_candidate: bool
    status_label: str
    orbital_period_days: float
    orbital_period_err: float
    transit_epoch_bjd: float
    transit_depth_pct: float
    transit_depth_ppm: float
    transit_duration_hours: float
    snr: float
    radius_ratio_rp_rstar: Optional[float] = None
    estimated_planet_radius_earth: Optional[float] = None
    estimated_semi_major_axis_au: Optional[float] = None
    estimated_equilibrium_temp_k: Optional[float] = None
    interpretation: str


class VisualizationData(BaseModel):
    # Raw light curve (downsampled to max ~3000 pts for performance)
    raw_time: List[float]
    raw_flux: List[float]

    # Processed / detrended light curve
    clean_time: List[float]
    detrended_flux: List[float]
    trend: List[float]

    # BLS periodogram
    bls_periods: List[float]
    bls_powers: List[float]
    bls_best_period: float
    bls_max_power: float

    # Phase-folded scatter
    folded_phase: List[float]
    folded_flux: List[float]

    # Global view (2001 bins)
    global_phase: List[float]
    global_flux: List[float]

    # Local transit view (201 bins)
    local_phase: List[float]
    local_flux: List[float]


class BLSStats(BaseModel):
    period_days: float
    transit_epoch_bjd: float
    duration_hours: float
    depth_pct: float
    depth_ppm: float
    snr: float
    max_power: float
    num_transits_observed: int


class AnalysisResponse(BaseModel):
    analysis_id: str
    tic_id: str
    data_source: str
    is_synthetic: bool
    status: str
    model_trained: bool
    classification: Optional[ClassificationResponse] = None
    transit_parameters: Optional[TransitParameters] = None
    bls_stats: Optional[BLSStats] = None
    preprocessing_summary: Optional[Dict[str, Any]] = None
    visualization: Optional[VisualizationData] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class AnalysisSummary(BaseModel):
    analysis_id: str
    tic_id: str
    status: str
    data_source: Optional[str] = None
    is_synthetic: Optional[bool] = None
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    orbital_period_days: Optional[float] = None
    snr: Optional[float] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class AnalysisListResponse(BaseModel):
    total: int
    items: List[AnalysisSummary]


class DashboardStats(BaseModel):
    total_analyses: int
    exoplanet_candidates: int
    eclipsing_binaries: int
    background_blends: int
    stellar_noise: int
    recent_analyses: List[AnalysisSummary]
