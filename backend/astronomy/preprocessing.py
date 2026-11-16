"""
preprocessing.py - Astronomical light curve conditioning, outlier rejection,
and baseline detrending with transit preservation for StarTrace.
"""

from typing import Tuple, Dict, Any
import numpy as np
from scipy.signal import savgol_filter


def condition_and_detrend_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    window_length: int = 101,
    polyorder: int = 2,
    sigma: float = 5.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Conditions and detrends raw astronomical time-series flux.

    Steps:
      1. Remove NaNs and infinities.
      2. Strictly sort chronologically.
      3. Median normalization (initial baseline ~ 1.0).
      4. 5-Sigma Outlier Clipping on positive spikes (cosmic rays) while preserving transit dips.
      5. Transit-protected Savitzky-Golay filtering to remove low-frequency stellar activity/trends.
      6. Final baseline normalization.

    Returns:
      (time_clean, flux_clean, trend, detrended_flux, summary_stats)
    """
    valid_mask = np.isfinite(time) & np.isfinite(flux)
    t = np.array(time[valid_mask], dtype=np.float64)
    f = np.array(flux[valid_mask], dtype=np.float64)

    if len(t) < 50:
        raise ValueError("Insufficient finite data points for light curve conditioning.")

    # Chronological sort
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    f = f[sort_idx]

    # Initial median normalization
    med_initial = np.median(f)
    if med_initial > 0:
        f = f / med_initial

    # Outlier clipping: Positive outliers (e.g. cosmic rays, flares) clipped at +sigma * std
    # Negative values below 0.05 are flagged as unphysical noise
    med = np.median(f)
    mad = np.median(np.abs(f - med))
    std = 1.4826 * mad
    std = max(std, 1e-6)

    # Allow deep negative dips (up to 30-50% for eclipsing binaries), reject positive noise spikes
    clip_mask = (f < (med + sigma * std)) & (f > 0.05)
    points_clipped = int(np.sum(~clip_mask))

    t = t[clip_mask]
    f = f[clip_mask]

    n_pts = len(f)
    if n_pts < 50:
        raise ValueError("Too few points remain after outlier rejection.")

    # Determine Savitzky-Golay window
    win = int(window_length)
    if win % 2 == 0:
        win += 1
    if win >= n_pts:
        win = n_pts - 1 if (n_pts - 1) % 2 != 0 else n_pts - 2
    win = max(win, polyorder + 3 if (polyorder + 3) % 2 != 0 else polyorder + 2)

    # Transit-Protected Detrending:
    # Identify candidate dip points below med - 2.8 * std, and interpolate them
    # before calculating the trend line so the filter does not flatten transit events.
    dip_threshold = med - 2.8 * std
    transit_candidate_pts = f < dip_threshold
    f_detrend_base = np.copy(f)

    if np.sum(~transit_candidate_pts) > win:
        f_detrend_base[transit_candidate_pts] = np.interp(
            t[transit_candidate_pts],
            t[~transit_candidate_pts],
            f[~transit_candidate_pts]
        )

    trend = savgol_filter(f_detrend_base, window_length=win, polyorder=polyorder)
    trend = np.where(trend <= 0, 1.0, trend)

    # Divide by stellar trend
    detrended = f / trend

    # Re-normalize median to 1.0
    baseline_median = np.median(detrended)
    if baseline_median > 0:
        detrended = detrended / baseline_median

    summary = {
        "initial_points": int(np.sum(valid_mask)),
        "clean_points": n_pts,
        "outliers_removed": points_clipped,
        "filter_window": win,
        "filter_polyorder": polyorder,
        "baseline_rms_ppm": float(np.std(detrended) * 1e6)
    }

    return t, f, trend, detrended, summary
