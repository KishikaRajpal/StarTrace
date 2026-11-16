"""
database/models.py - SQLAlchemy ORM models for StarTrace PostgreSQL database.
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, Text, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from .session import Base


def gen_uuid():
    return str(uuid.uuid4())


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(String, primary_key=True, default=gen_uuid)
    tic_id = Column(String(64), nullable=False, index=True)
    data_source = Column(String(256))
    is_synthetic = Column(Boolean, default=True)
    status = Column(String(32), default="pending")  # pending | processing | completed | failed
    error_message = Column(Text, nullable=True)

    # Classification result
    predicted_class = Column(String(64), nullable=True)
    class_id = Column(Integer, nullable=True)
    confidence_score = Column(Float, nullable=True)

    # Transit parameters
    orbital_period_days = Column(Float, nullable=True)
    transit_depth_pct = Column(Float, nullable=True)
    transit_depth_ppm = Column(Float, nullable=True)
    transit_duration_hours = Column(Float, nullable=True)
    transit_epoch_bjd = Column(Float, nullable=True)
    snr = Column(Float, nullable=True)
    max_bls_power = Column(Float, nullable=True)
    num_transits = Column(Integer, nullable=True)

    # Derived planetary parameters (only valid for class 0)
    planet_radius_earth = Column(Float, nullable=True)
    semi_major_axis_au = Column(Float, nullable=True)
    equilibrium_temp_k = Column(Float, nullable=True)

    # Storage reference paths
    data_file_path = Column(String(512), nullable=True)   # Compressed npz with time/flux arrays
    result_file_path = Column(String(512), nullable=True) # Full analysis JSON

    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    classification = relationship("ClassificationResult", back_populates="analysis", uselist=False, cascade="all, delete-orphan")


class ClassificationResult(Base):
    __tablename__ = "classification_results"

    id = Column(String, primary_key=True, default=gen_uuid)
    analysis_id = Column(String, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True)

    prob_exoplanet_transit = Column(Float, nullable=True)
    prob_eclipsing_binary = Column(Float, nullable=True)
    prob_stellar_blend = Column(Float, nullable=True)
    prob_stellar_noise = Column(Float, nullable=True)

    analysis = relationship("Analysis", back_populates="classification")


class ModelMetrics(Base):
    __tablename__ = "model_metrics"

    id = Column(String, primary_key=True, default=gen_uuid)
    model_name = Column(String(128), nullable=False)
    accuracy = Column(Float, nullable=True)
    macro_precision = Column(Float, nullable=True)
    macro_recall = Column(Float, nullable=True)
    macro_f1 = Column(Float, nullable=True)
    total_test_samples = Column(Integer, nullable=True)
    notes = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow)
