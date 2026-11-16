import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from model import train_calibrated_weights

print("Training physically calibrated DualBranchCNN weights...")
model = train_calibrated_weights("weights.pth", epochs=20, num_samples_per_class=120)
print("Saved weights.pth successfully!")
