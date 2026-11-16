"""
parameter_estimation.py - Astronomical parameter estimation, transit modeling,
and physical characterization of potential exoplanet candidates for StarTrace.
"""

from typing import Dict, Any
import numpy as np


def estimate_transit_parameters(
    bls_stats: Dict[str, Any],
    predicted_class: str,
    confidence: float
) -> Dict[str, Any]:
    """
    Extracts and computes astrophysical transit parameters.
    Clearly marks candidates as 'Potential Exoplanet Candidate' only when
    classified as an exoplanet transit signal.
    """
    period = float(bls_stats.get("period_days", 0.0))
    raw_depth = float(bls_stats.get("raw_depth", 0.0))
    depth_pct = float(bls_stats.get("depth_pct", 0.0))
    depth_ppm = float(bls_stats.get("depth_ppm", 0.0))
    duration_hr = float(bls_stats.get("duration_hours", 0.0))
    epoch_bjd = float(bls_stats.get("transit_epoch_bjd", 0.0))
    snr = float(bls_stats.get("snr", 0.0))

    is_exoplanet = (predicted_class == "Exoplanet Transit")

    # Physical estimates assuming nominal Solar/M-dwarf host star parameters
    # Radius of Sun ~ 109.2 Earth radii
    # Rp/R* = sqrt(depth)
    radius_ratio = float(np.sqrt(max(0.0, raw_depth)))
    r_earth_estimate = float(radius_ratio * 109.2)

    # Semi-major axis (Kepler's Third Law for 1 Solar mass host)
    # a = (P / 365.25)^(2/3) AU
    semi_major_axis_au = float((period / 365.25)**(2.0 / 3.0)) if period > 0 else 0.0

    # Equilibrium temperature estimate (assuming albedo 0.3, solar irradiance)
    # Teq ~ 278 * (1 AU / a)^0.5 K
    if semi_major_axis_au > 0:
        t_eq_k = float(278.0 / np.sqrt(semi_major_axis_au))
    else:
        t_eq_k = 0.0

    # Ingress/egress estimated time fraction (typical trapezoid ~ 15-20% of duration)
    tau_ingress_hr = float(duration_hr * 0.18)

    # Uncertainty approximations (1/SNR scaled)
    sigma_scale = max(0.02, min(0.30, 2.5 / (snr + 1e-4)))
    period_err = round(period * sigma_scale * 0.005, 5)
    depth_err_pct = round(depth_pct * sigma_scale, 4)
    duration_err_hr = round(duration_hr * sigma_scale, 2)

    if is_exoplanet:
        classification_status = "Potential Exoplanet Candidate"
        interpretation_note = (
            f"Periodic U-shaped occultation detected with period {period:.4f} days and depth {depth_pct:.3f}%. "
            f"Fitted transit characteristics are consistent with a planetary companion of approximately "
            f"{r_earth_estimate:.2f} R_Earth at an orbital distance of {semi_major_axis_au:.3f} AU."
        )
    elif predicted_class == "Eclipsing Binary":
        classification_status = "Eclipsing Binary System"
        interpretation_note = (
            f"Deep periodic transit with depth {depth_pct:.2f}% and secondary minima. "
            "Characteristics are typical of a stellar binary eclipse rather than a planetary occultation. "
            "Planetary companion parameters are not applicable."
        )
    elif predicted_class == "Background Blend":
        classification_status = "Suspected Background Blend"
        interpretation_note = (
            f"Periodic signal with depth {depth_pct:.3f}% exhibiting transit asymmetry. "
            "Likely caused by flux contamination from an unresolved off-target eclipsing binary in the TESS aperture. "
            "Planetary companion status is rejected."
        )
    else:
        classification_status = "Non-Transit Stellar Activity / Noise"
        interpretation_note = (
            "Detected periodicity is dominated by quasi-periodic stellar rotational starspot modulation, "
            "magnetic flares, or instrumental systematic drift. No confirmed transit ingress/egress profile."
        )

    return {
        "is_candidate": is_exoplanet,
        "status_label": classification_status,
        "orbital_period_days": period,
        "orbital_period_err": period_err,
        "transit_epoch_bjd": epoch_bjd,
        "transit_depth_pct": depth_pct,
        "transit_depth_ppm": depth_ppm,
        "transit_depth_err_pct": depth_err_pct,
        "transit_duration_hours": duration_hr,
        "transit_duration_err_hours": duration_err_hr,
        "snr": snr,
        "radius_ratio_rp_rstar": round(radius_ratio, 4),
        "estimated_planet_radius_earth": round(r_earth_estimate, 2) if is_exoplanet else None,
        "estimated_semi_major_axis_au": round(semi_major_axis_au, 4) if is_exoplanet else None,
        "estimated_equilibrium_temp_k": round(t_eq_k, 1) if is_exoplanet else None,
        "ingress_egress_duration_hours": round(tau_ingress_hr, 2) if is_exoplanet else None,
        "interpretation": interpretation_note
    }
