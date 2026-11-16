"""
phase_folding.py - Phase folding and dual-branch (Global & Local) feature preparation
for StarTrace.
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy.stats import binned_statistic


def compute_phase_and_dual_views(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    global_bins: int = 2001,
    local_bins: int = 201,
    local_phase_width: float = 0.10,
    max_scatter_points: int = 2500
) -> Dict[str, Any]:
    """
    Phase-folds the light curve centered on transit epoch t0 into [-0.5, 0.5].
    Computes:
      1. Global View (2001 bins) spanning [-0.5, 0.5]
      2. Local View (201 bins) zoomed on transit window [-width/2, +width/2]
      3. Downsampled raw phase-folded scatter for interactive Plotly charts.
    """
    # Phase calculation centered on transit at phase 0.0
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5

    def bin_phase_series(
        p_arr: np.ndarray,
        f_arr: np.ndarray,
        num_bins: int,
        p_min: float,
        p_max: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        bin_edges = np.linspace(p_min, p_max, num_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        in_range = (p_arr >= p_min) & (p_arr <= p_max)
        sub_p = p_arr[in_range]
        sub_f = f_arr[in_range]

        if len(sub_p) == 0:
            return bin_centers, np.ones(num_bins, dtype=np.float32)

        stat, _, _ = binned_statistic(sub_p, sub_f, statistic="median", bins=bin_edges)

        # Handle any empty bins via interpolation
        nan_mask = np.isnan(stat)
        if np.all(nan_mask):
            stat = np.ones(num_bins, dtype=np.float32)
        elif np.any(nan_mask):
            valid_x = bin_centers[~nan_mask]
            valid_y = stat[~nan_mask]
            stat[nan_mask] = np.interp(bin_centers[nan_mask], valid_x, valid_y)

        return bin_centers, stat.astype(np.float32)

    half_w = local_phase_width / 2.0
    centers_global, global_view = bin_phase_series(phase, flux, global_bins, -0.5, 0.5)
    centers_local, local_view = bin_phase_series(phase, flux, local_bins, -half_w, half_w)

    # Downsample raw phase-folded points for responsive frontend visualization
    n_pts = len(phase)
    if n_pts > max_scatter_points:
        step = max(1, n_pts // max_scatter_points)
        sort_order = np.argsort(phase)
        phase_sorted = phase[sort_order][::step]
        flux_sorted = flux[sort_order][::step]
    else:
        sort_order = np.argsort(phase)
        phase_sorted = phase[sort_order]
        flux_sorted = flux[sort_order]

    # Filter local view scatter
    local_mask = (phase_sorted >= -half_w) & (phase_sorted <= half_w)

    return {
        "global_view": global_view,
        "centers_global": centers_global,
        "local_view": local_view,
        "centers_local": centers_local,
        "folded_phase_scatter": phase_sorted.tolist(),
        "folded_flux_scatter": flux_sorted.tolist(),
        "local_phase_scatter": phase_sorted[local_mask].tolist(),
        "local_flux_scatter": flux_sorted[local_mask].tolist(),
        "local_phase_width": local_phase_width
    }
