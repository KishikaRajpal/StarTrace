import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from pipeline import process_lightcurve, PRESET_METADATA
from model import get_model, classify_lightcurve

model = get_model("weights.pth")

print("\n--- Testing All 4 Presets ---")
for tic_id, meta in PRESET_METADATA.items():
    res = process_lightcurve(tic_id=tic_id)
    clf = classify_lightcurve(model, res["dual_views"]["global_view"], res["dual_views"]["local_view"], bls_stats=res["bls_stats"])
    print(f"TIC {tic_id} ({meta['name']}):")
    print(f"  Target Category : {meta['category']}")
    print(f"  Predicted Class : {clf['prediction']} ({clf['confidence']*100:.1f}%)")
    print(f"  BLS Stats       : P={res['bls_stats']['period_days']:.3f}d, depth={res['bls_stats']['depth_pct']:.3f}%, SNR={res['bls_stats']['snr']:.1f}")
