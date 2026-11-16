import numpy as np
from pipeline import process_lightcurve, PRESET_METADATA

for tic_id, meta in PRESET_METADATA.items():
    res = process_lightcurve(tic_id=tic_id)
    gv = res["dual_views"]["global_view"]
    lv = res["dual_views"]["local_view"]
    bls = res["bls_stats"]

    center = len(lv) // 2
    local_dip = 1.0 - np.min(lv[center - 20 : center + 20])
    out_rms = np.std(np.concatenate([gv[200:850], gv[1150:1800]]))
    sec_dip = 1.0 - min(np.min(gv[:100]), np.min(gv[-100:]))
    
    left = lv[center - 30 : center]
    right = lv[center : center + 30][::-1]
    asym = np.mean(np.abs(left - right)) / (local_dip + 1e-6)
    var_ratio = out_rms / (local_dip + 1e-6)

    print(f"\nTIC {tic_id} ({meta['category']}):")
    print(f"  local_dip: {local_dip*100:.3f}%")
    print(f"  depth_pct: {bls['depth_pct']:.3f}%")
    print(f"  sec_dip:   {sec_dip*100:.3f}%")
    print(f"  out_rms:   {out_rms*100:.4f}%")
    print(f"  var_ratio: {var_ratio:.3f}")
    print(f"  asym:      {asym:.3f}")
