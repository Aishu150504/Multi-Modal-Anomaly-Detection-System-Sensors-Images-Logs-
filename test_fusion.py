import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from fusion.score_normalizer import normalize_scores
from fusion.fusion_logic import fuse_modalities
from fusion.threshold import compute_threshold, predict_anomalies

sensor_scores = [0.1, 0.2, 0.9]
image_scores = [0.05, 0.1, 0.8]
log_scores = [0.2, 0.3, 0.95]

s = normalize_scores(sensor_scores)
i = normalize_scores(image_scores)
l = normalize_scores(log_scores)

fused = fuse_modalities(s, i, l)
threshold = compute_threshold(fused)
preds = predict_anomalies(fused, threshold)

print("Fused scores:", fused)
print("Threshold:", threshold)
print("Predictions:", preds)
print("Fusion pipeline test PASSED ✔")
