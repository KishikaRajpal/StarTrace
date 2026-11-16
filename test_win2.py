import numpy as np
from scipy.signal import savgol_filter
from pipeline import generate_synthetic_lightcurve, run_bls_search

t, f, _ = generate_synthetic_lightcurve("25155310")
med = np.median(f)
std = 1.4826 * np.median(np.abs(f - med))

win = 101
trend1 = savgol_filter(f, window_length=win, polyorder=2)
res = f - trend1
mask = res > -2.5 * std
f_interp = np.copy(f)
f_interp[~mask] = np.interp(t[~mask], t[mask], f[mask])
trend = savgol_filter(f_interp, window_length=win, polyorder=2)

detrended = f / trend
detrended = detrended / np.median(detrended)

bls, _ = run_bls_search(t, detrended)
print(f"Iterative Detrending Win 101: Period={bls['period_days']:.3f}, Depth={bls['depth_pct']:.4f}%, SNR={bls['snr']:.2f}")
