"""
pipeline.py - NASA TESS Light Curve Ingestion, Conditioning, BLS Periodicity Search,
and Dual-View Binning Pipeline for StarTrace.
"""

import hashlib
import warnings
import socket
import concurrent.futures
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic

# Set global socket timeout to prevent any network hanging on MAST queries
socket.setdefaulttimeout(5.0)

# Suppress minor Astropy/Lightkurve warnings during processing
warnings.filterwarnings("ignore")

PRESET_METADATA = {
    "25155310": {
        "name": "Kepler-1649c",
        "category": "Exoplanet Transit",
        "description": "Confirmed Earth-sized exoplanet orbiting in habitable zone of M-dwarf star.",
        "expected_period": 8.689,
        "expected_depth_pct": 0.22,
        "expected_duration_hr": 3.2,
    },
    "149603524": {
        "name": "Eclipsing Binary System",
        "category": "Eclipsing Binary",
        "description": "Detached double-lined eclipsing binary exhibiting deep primary and secondary minima.",
        "expected_period": 2.342,
        "expected_depth_pct": 8.5,
        "expected_duration_hr": 4.5,
    },
    "38846515": {
        "name": "Background Blend Source",
        "category": "Background Blend",
        "description": "Nearby background eclipsing binary contaminating the photometric aperture.",
        "expected_period": 3.754,
        "expected_depth_pct": 0.85,
        "expected_duration_hr": 3.2,
    },
    "261136638": {
        "name": "Stellar Noise / Variable Star",
        "category": "Stellar Noise / Activity",
        "description": "Rapidly rotating active star exhibiting rotational starspot modulation and flares.",
        "expected_period": 1.428,
        "expected_depth_pct": 0.55,
        "expected_duration_hr": 2.1,
    }
}


def generate_synthetic_lightcurve(tic_id="25155310", cadence_min=2.0, duration_days=27.4):
    """
    Deterministic synthetic light-curve generator fallback.
    Ensures zero crashes during live demos if NASA MAST servers are slow or unreachable.
    """
    tic_str = str(tic_id).strip().replace("TIC", "").replace("tic", "").strip()
    preset = PRESET_METADATA.get(tic_str)

    # Deterministic seed based on TIC ID
    seed = int(hashlib.md5(tic_str.encode()).hexdigest()[:8], 16) % (2**31)
    rng = np.random.RandomState(seed)

    # Time series over ~27.4 days (1 standard TESS sector) at 2-minute cadence (~19,728 pts)
    cadence_day = cadence_min / (24.0 * 60.0)
    time = np.arange(0.0, duration_days, cadence_day)
    n_pts = len(time)

    # White photometric noise baseline (~250 ppm)
    noise = rng.normal(0, 0.00025, size=n_pts)
    flux = np.ones(n_pts) + noise

    # Add low-frequency stellar activity baseline
    stellar_activity_period = rng.uniform(8.0, 16.0)
    stellar_activity = 0.0008 * np.sin(2 * np.pi * time / stellar_activity_period + rng.uniform(0, 2*np.pi))
    flux += stellar_activity

    if preset and preset["category"] == "Exoplanet Transit":
        period = preset["expected_period"]
        t0 = 1.2
        duration = preset["expected_duration_hr"] / 24.0
        depth = preset["expected_depth_pct"] / 100.0

        phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
        in_transit = np.abs(phase * period) < (duration / 2.0)
        z = np.abs((phase * period) / (duration / 2.0))
        transit_shape = np.clip(1.0 - z**4, 0, 1)
        flux[in_transit] -= depth * transit_shape[in_transit]

    elif preset and preset["category"] == "Eclipsing Binary":
        period = preset["expected_period"]
        t0 = 0.8
        duration = preset["expected_duration_hr"] / 24.0
        depth_pri = preset["expected_depth_pct"] / 100.0
        depth_sec = depth_pri * 0.45

        # Primary eclipse at phase 0 (deep V-shape)
        phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
        in_pri = np.abs(phase * period) < (duration / 2.0)
        v_pri = np.maximum(0.0, 1.0 - np.abs((phase * period) / (duration / 2.0)))
        flux[in_pri] -= depth_pri * v_pri[in_pri]

        # Secondary eclipse at phase 0.5
        phase_sec = ((time - t0) % period) / period - 0.5
        in_sec = np.abs(phase_sec * period) < (duration / 2.0)
        v_sec = np.maximum(0.0, 1.0 - np.abs((phase_sec * period) / (duration / 2.0)))
        flux[in_sec] -= depth_sec * v_sec[in_sec]

        # Out-of-eclipse ellipsoidal variation
        flux += 0.003 * np.cos(4 * np.pi * (time - t0) / period)

    elif preset and preset["category"] == "Background Blend":
        period = preset["expected_period"]
        t0 = 1.5
        duration = preset["expected_duration_hr"] / 24.0
        depth = preset["expected_depth_pct"] / 100.0

        phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
        in_transit = np.abs(phase * period) < (duration / 2.0)
        v_blend = np.maximum(0.0, 1.0 - np.abs(((phase * period) + 0.012) / (duration / 2.0))**1.5)
        flux[in_transit] -= depth * v_blend[in_transit]

    elif preset and preset["category"] == "Stellar Noise / Activity":
        var_period = preset["expected_period"]
        flux += 0.006 * np.sin(2 * np.pi * time / var_period)
        flux += 0.003 * np.cos(4 * np.pi * time / var_period + 0.5)
        flare_locs = rng.choice(n_pts, size=4, replace=False)
        for loc in flare_locs:
            decay_window = min(120, n_pts - loc)
            decay = np.exp(-np.linspace(0, 4, decay_window))
            flux[loc:loc+decay_window] += 0.008 * decay

    else:
        period = rng.uniform(2.5, 7.5)
        t0 = 1.0
        duration = rng.uniform(2.0, 4.0) / 24.0
        depth = rng.uniform(0.002, 0.008)
        phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
        in_transit = np.abs(phase * period) < (duration / 2.0)
        transit_shape = np.clip(1.0 - (np.abs(phase * period) / (duration / 2.0))**3, 0, 1)
        flux[in_transit] -= depth * transit_shape[in_transit]

    return time, flux, True


def _mast_download_worker(tic_clean, author_pref):
    """Worker function for MAST querying."""
    import lightkurve as lk
    try:
        from astroquery.mast import Observations
        Observations.TIMEOUT = 5
    except Exception:
        pass

    search = lk.search_lightcurve(f"TIC {tic_clean}", mission="TESS", author=author_pref)
    if len(search) == 0:
        search = lk.search_lightcurve(f"TIC {tic_clean}", mission="TESS")

    if len(search) > 0:
        lc = search[0].download()
        if lc is not None:
            time = np.asarray(lc.time.value, dtype=np.float64)
            if hasattr(lc, "pdcsap_flux") and lc.pdcsap_flux is not None and not np.all(np.isnan(lc.pdcsap_flux.value)):
                flux = np.asarray(lc.pdcsap_flux.value, dtype=np.float64)
            else:
                flux = np.asarray(lc.flux.value, dtype=np.float64)

            valid = np.isfinite(time) & np.isfinite(flux)
            time = time[valid]
            flux = flux[valid]

            if len(time) > 200:
                return time, flux
    return None


def fetch_tess_lightcurve(tic_id, author_pref="SPOC", timeout_sec=5):
    """
    Live streaming ingestion of TESS light curves using Lightkurve.
    Queries NASA MAST with an asynchronous timeout.
    Falls back to deterministic synthetic generator on network error or missing data.
    """
    tic_clean = str(tic_id).strip().replace("TIC", "").replace("tic", "").strip()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_mast_download_worker, tic_clean, author_pref)
        res = future.result(timeout=timeout_sec)
        executor.shutdown(wait=False, cancel_futures=True)
        if res is not None:
            time, flux = res
            return time, flux, False
    except Exception:
        executor.shutdown(wait=False, cancel_futures=True)

    time, flux, _ = generate_synthetic_lightcurve(tic_clean)
    return time, flux, True


def condition_lightcurve(time, flux, window_length=101, polyorder=2, sigma=5.0):
    """
    Drop NaNs, apply 5-sigma outlier rejection, and detrend low-frequency
    stellar variability using Savitzky-Golay filtering. Baseline flux normalized to 1.0.
    Protects negative transit excursions from clipping or flattening.
    """
    valid_mask = np.isfinite(time) & np.isfinite(flux)
    t = np.array(time[valid_mask], dtype=np.float64)
    f = np.array(flux[valid_mask], dtype=np.float64)

    order = np.argsort(t)
    t = t[order]
    f = f[order]

    med_f = np.median(f)
    if med_f > 0:
        f = f / med_f

    # 5-Sigma Outlier Clipping on positive spikes (cosmic rays); preserve negative dips
    med = np.median(f)
    std = 1.4826 * np.median(np.abs(f - med))
    std = max(std, 1e-6)

    clip_mask = (f < (med + sigma * std)) & (f > 0.05)
    t = t[clip_mask]
    f = f[clip_mask]

    # Savitzky-Golay Detrending with transit preservation
    n_pts = len(f)
    win = int(window_length)
    if win % 2 == 0:
        win += 1
    if win >= n_pts:
        win = n_pts - 1 if (n_pts - 1) % 2 != 0 else n_pts - 2
    win = max(win, polyorder + 2 if (polyorder + 2) % 2 != 0 else polyorder + 3)

    # Detrend on out-of-transit stellar baseline
    f_clean = np.copy(f)
    dip_threshold = med - 2.8 * std
    transit_pts = f < dip_threshold
    if np.sum(~transit_pts) > win:
        f_clean[transit_pts] = np.interp(t[transit_pts], t[~transit_pts], f[~transit_pts])
    trend = savgol_filter(f_clean, window_length=win, polyorder=polyorder)

    trend = np.where(trend <= 0, 1.0, trend)
    detrended = f / trend
    baseline_median = np.median(detrended)
    detrended = detrended / baseline_median

    return t, f, trend, detrended


def run_bls_search(time, flux, min_period=0.5, max_period=15.0, num_durations=8):
    """
    Runs Astropy Box Least Squares (BLS) periodogram search.
    Extracts Orbital Period, Transit Epoch (t0), Duration, Depth, and SNR.
    """
    from astropy.timeseries import BoxLeastSquares

    durations = np.linspace(1.2 / 24.0, 7.0 / 24.0, num_durations)

    bls = BoxLeastSquares(time, flux)
    periodogram = bls.autopower(
        durations,
        minimum_period=min_period,
        maximum_period=max_period,
        frequency_factor=1.2
    )

    best_idx = np.argmax(periodogram.power)
    best_period = float(periodogram.period[best_idx])
    best_t0 = float(periodogram.transit_time[best_idx])
    best_duration = float(periodogram.duration[best_idx])
    raw_box_depth = float(periodogram.depth[best_idx])
    max_power = float(periodogram.power[best_idx])

    duration_hours = best_duration * 24.0

    # Calculate exact photometric transit depth
    phase = ((time - best_t0 + 0.5 * best_period) % best_period) / best_period - 0.5
    in_transit = np.abs(phase * best_period) < (best_duration / 2.0)
    n_in_transit = np.sum(in_transit)

    if n_in_transit > 8:
        out_flux = np.median(flux[~in_transit])
        in_flux = np.percentile(flux[in_transit], 25)
        depth = float(max(0.0002, out_flux - in_flux))
    else:
        depth = float(max(0.0002, raw_box_depth))

    depth_pct = depth * 100.0

    baseline_std = 1.4826 * np.median(np.abs(flux - np.median(flux)))
    baseline_std = max(baseline_std, 1e-6)

    total_time = time[-1] - time[0]
    n_transits = max(1.0, total_time / best_period)
    cadence_day = np.median(np.diff(time))
    pts_per_transit = max(3.0, best_duration / cadence_day)
    total_transit_pts = pts_per_transit * n_transits

    snr = float((depth / baseline_std) * np.sqrt(total_transit_pts))
    snr = max(snr, max_power * 1000.0)

    bls_stats = {
        "period_days": best_period,
        "t0": best_t0,
        "duration_hours": duration_hours,
        "duration_days": best_duration,
        "depth_pct": depth_pct,
        "raw_depth": depth,
        "snr": snr,
        "max_power": max_power
    }

    periodogram_data = {
        "periods": np.asarray(periodogram.period, dtype=np.float32),
        "powers": np.asarray(periodogram.power, dtype=np.float32)
    }

    return bls_stats, periodogram_data


def compute_dual_views(time, flux, period, t0):
    """
    Phase-folds the light curve centered on t0 into [-0.5, 0.5]
    and computes median-binned fixed-length arrays:
      - Global View: 2001 bins over [-0.5, 0.5]
      - Local View: 201 bins zoomed over [-0.05, 0.05]
    """
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5

    def bin_phase(p, f, num_bins, p_min, p_max):
        bin_edges = np.linspace(p_min, p_max, num_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        mask = (p >= p_min) & (p <= p_max)
        sub_p = p[mask]
        sub_f = f[mask]

        if len(sub_p) == 0:
            return bin_centers, np.ones(num_bins, dtype=np.float32)

        stat, _, _ = binned_statistic(sub_p, sub_f, statistic="median", bins=bin_edges)

        nans = np.isnan(stat)
        if np.all(nans):
            stat = np.ones(num_bins, dtype=np.float32)
        elif np.any(nans):
            valid_x = bin_centers[~nans]
            valid_y = stat[~nans]
            stat[nans] = np.interp(bin_centers[nans], valid_x, valid_y)

        return bin_centers, stat.astype(np.float32)

    centers_global, global_view = bin_phase(phase, flux, 2001, -0.5, 0.5)
    centers_local, local_view = bin_phase(phase, flux, 201, -0.05, 0.05)

    return {
        "phase_raw": phase,
        "flux_raw": flux,
        "centers_global": centers_global,
        "global_view": global_view,
        "centers_local": centers_local,
        "local_view": local_view
    }


def process_lightcurve(
    source_type="NASA MAST Live Stream",
    tic_id="25155310",
    csv_file=None,
    window_length=101,
    min_period=0.5,
    max_period=15.0
):
    """
    End-to-end processing pipeline orchestrator.
    Handles data ingestion, conditioning, BLS periodicity search, and dual-view binning.
    """
    is_fallback = False
    source_label = ""

    if source_type == "Upload CSV" and csv_file is not None:
        try:
            df = pd.read_csv(csv_file)
            cols = [c.lower().strip() for c in df.columns]
            t_col = None
            f_col = None
            for col, orig in zip(cols, df.columns):
                if col in ["time", "bjd", "btjd", "t", "date"]:
                    t_col = orig
                elif col in ["flux", "pdcsap_flux", "sap_flux", "f", "mag", "intensity"]:
                    f_col = orig

            if t_col is None:
                t_col = df.columns[0]
            if f_col is None:
                f_col = df.columns[1]

            time = np.asarray(df[t_col].values, dtype=np.float64)
            raw_flux = np.asarray(df[f_col].values, dtype=np.float64)
            source_label = f"Uploaded CSV ({getattr(csv_file, 'name', 'photometry.csv')})"
        except Exception:
            time, raw_flux, is_fallback = generate_synthetic_lightcurve("25155310")
            source_label = "CSV Parse Fallback (Kepler-1649c)"
    else:
        tic_str = str(tic_id).strip().replace("TIC", "").replace("tic", "").strip()
        time, raw_flux, is_fallback = fetch_tess_lightcurve(tic_str, timeout_sec=5)
        source_label = f"TIC {tic_str} ({'Synthetic Fallback' if is_fallback else 'NASA MAST Live Stream'})"

    # Step 2: Conditioning & Detrending
    time_clean, flux_clean, trend, detrended_flux = condition_lightcurve(
        time, raw_flux, window_length=window_length, polyorder=2, sigma=5.0
    )

    # Step 3: BLS Periodicity Search
    bls_stats, periodogram_data = run_bls_search(
        time_clean, detrended_flux, min_period=min_period, max_period=max_period
    )

    # Step 4: Dual-View Binning
    dual_views = compute_dual_views(
        time_clean, detrended_flux, bls_stats["period_days"], bls_stats["t0"]
    )

    return {
        "time": time_clean,
        "raw_flux": flux_clean,
        "trend": trend,
        "detrended_flux": detrended_flux,
        "bls_stats": bls_stats,
        "periodogram": periodogram_data,
        "dual_views": dual_views,
        "is_fallback": is_fallback,
        "source_label": source_label,
        "tic_id": str(tic_id).strip()
    }
