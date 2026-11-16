import numpy as np
from pipeline import generate_synthetic_lightcurve, condition_lightcurve, run_bls_search

t, f, _ = generate_synthetic_lightcurve("25155310")
for win in [51, 101, 151, 201, 301]:
    t_c, f_c, tr, d = condition_lightcurve(t, f, window_length=win)
    bls, _ = run_bls_search(t_c, d)
    print(f"Win {win}: Period={bls['period_days']:.3f}, Depth={bls['depth_pct']:.4f}%, SNR={bls['snr']:.2f}")
