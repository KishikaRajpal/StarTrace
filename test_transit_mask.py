import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from pipeline import generate_synthetic_lightcurve, run_bls_search

t, f, _ = generate_synthetic_lightcurve("25155310")

# Iterative SavGol with transit protection
win = 101
trend = savgol_filter(f, win, 2)
for _ in range(3):
    res = f - trend
    std = 1.4826 * np.median(np.abs(res - np.median(res)))
    # Transits are negative dips: mask points below -1.5 std
    mask = res > -1.5 * std
    if np.sum(~mask) > 0:
        f_int = np.interp(t, t[mask], f[mask])
        trend = savgol_filter(f_int, win, 2)

detr = f / trend
detr = detr / np.median(detr)
bls, _ = run_bls_search(t, detr)
print("Detected depth %:", bls["depth_pct"], "SNR:", bls["snr"])
