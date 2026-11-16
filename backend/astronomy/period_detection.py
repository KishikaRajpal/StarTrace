"""
period_detection.py - Box Least Squares (BLS) periodic transit signal search
and power spectrum analysis for StarTrace.
"""

from typing import Tuple, Dict, Any
import numpy as np


def run_bls_detection(
    time: np.ndarray,
    flux: np.ndarray,
    min_period: float = 0.5,
    max_period: float = 15.0,
    num_durations: int = 8,
    frequency_factor: float = 1.2,
    max_spectrum_points: int = 800
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Executes Astropy Box Least Squares (BLS) periodic transit search.

    Extracts:
      - Best period (days)
      - Transit Epoch t0 (BJD)
      - Transit duration (hours / days)
      - Transit depth (% and raw fraction)
      - Max BLS Power
      - Signal-to-Noise Ratio (SNR)
      - Downsampled periodogram spectrum for interactive frontend graphing.
    """
    from astropy.timeseries import BoxLeastSquares

    total_time_span = float(time[-1] - time[0])
    effective_max_period = min(max_period, max(1.0, total_time_span / 2.1))
    effective_min_period = max(0.2, min_period)

    durations = np.linspace(1.2 / 24.0, min(7.0 / 24.0, effective_min_period * 0.25), num_durations)

    bls = BoxLeastSquares(time, flux)
    periodogram = bls.autopower(
        durations,
        minimum_period=effective_min_period,
        maximum_period=effective_max_period,
        frequency_factor=frequency_factor
    )

    if len(periodogram.power) == 0 or np.all(np.isnan(periodogram.power)):
        raise ValueError("BLS search yielded empty or invalid power spectrum.")

    best_idx = int(np.argmax(periodogram.power))
    best_period = float(periodogram.period[best_idx])
    best_t0 = float(periodogram.transit_time[best_idx])
    best_duration_days = float(periodogram.duration[best_idx])
    raw_box_depth = float(periodogram.depth[best_idx])
    max_power = float(periodogram.power[best_idx])

    duration_hours = best_duration_days * 24.0

    # Calculate actual photometric depth in-transit vs out-of-transit
    phase = ((time - best_t0 + 0.5 * best_period) % best_period) / best_period - 0.5
    in_transit_mask = np.abs(phase * best_period) < (best_duration_days / 2.0)
    n_in_transit = int(np.sum(in_transit_mask))

    if n_in_transit >= 6:
        out_flux = float(np.median(flux[~in_transit_mask]))
        in_flux = float(np.percentile(flux[in_transit_mask], 25))
        photometric_depth = max(0.0001, out_flux - in_flux)
    else:
        photometric_depth = max(0.0001, raw_box_depth)

    depth_pct = photometric_depth * 100.0
    depth_ppm = photometric_depth * 1e6

    # Robust baseline standard deviation via MAD
    baseline_std = 1.4826 * np.median(np.abs(flux - np.median(flux)))
    baseline_std = max(float(baseline_std), 1e-6)

    # Estimate total number of transits and SNR
    n_transits = max(1.0, total_time_span / best_period)
    cadence_day = float(np.median(np.diff(time)))
    pts_per_transit = max(3.0, best_duration_days / (cadence_day + 1e-8))
    total_transit_points = pts_per_transit * n_transits

    snr = float((photometric_depth / baseline_std) * np.sqrt(total_transit_points))
    # Enforce minimum physical lower bound based on BLS power
    snr = max(snr, max_power * 800.0)

    stats = {
        "period_days": round(best_period, 5),
        "transit_epoch_bjd": round(best_t0, 5),
        "duration_hours": round(duration_hours, 3),
        "duration_days": round(best_duration_days, 5),
        "depth_pct": round(depth_pct, 4),
        "depth_ppm": round(depth_ppm, 1),
        "raw_depth": float(photometric_depth),
        "snr": round(snr, 2),
        "max_power": round(max_power, 4),
        "num_transits_observed": int(round(n_transits)),
        "in_transit_points": n_in_transit
    }

    # Downsample periodogram for fast frontend Plotly rendering
    periods_arr = np.asarray(periodogram.period, dtype=np.float32)
    powers_arr = np.asarray(periodogram.power, dtype=np.float32)

    total_spec = len(periods_arr)
    if total_spec > max_spectrum_points:
        step = max(1, total_spec // max_spectrum_points)
        periods_down = periods_arr[::step].tolist()
        powers_down = powers_arr[::step].tolist()
        # Always include the peak
        if best_period not in periods_down:
            periods_down.append(best_period)
            powers_down.append(max_power)
            # Re-sort
            sort_k = np.argsort(periods_down)
            periods_down = [periods_down[i] for i in sort_k]
            powers_down = [powers_down[i] for i in sort_k]
    else:
        periods_down = periods_arr.tolist()
        powers_down = powers_arr.tolist()

    periodogram_summary = {
        "periods": periods_down,
        "powers": powers_down,
        "best_period": stats["period_days"],
        "max_power": stats["max_power"]
    }

    return stats, periodogram_summary
