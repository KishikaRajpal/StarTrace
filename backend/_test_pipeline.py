import sys, os
sys.path.insert(0, '.')
from astronomy.data_loader import fetch_tess_lightcurve
from astronomy.preprocessing import condition_and_detrend_lightcurve
from astronomy.period_detection import run_bls_detection
from astronomy.phase_folding import compute_phase_and_dual_views
from astronomy.parameter_estimation import estimate_transit_parameters
from ml.inference import load_model, classify_lightcurve, get_model_status
from app.database.session import init_db

print('All imports OK')

t, f, meta = fetch_tess_lightcurve('25155310', timeout_sec=3)
t2, f2, trend, det, pp = condition_and_detrend_lightcurve(t, f)
bls, pg = run_bls_detection(t2, det)
views = compute_phase_and_dual_views(t2, det, bls['period_days'], bls['transit_epoch_bjd'])
load_model()
clf = classify_lightcurve(views['global_view'], views['local_view'], bls)
print('Pipeline OK:', clf['predicted_class'], str(clf['confidence_pct']) + '%')
print('DB init...')
init_db()
print('DB init OK - all systems check passed')
