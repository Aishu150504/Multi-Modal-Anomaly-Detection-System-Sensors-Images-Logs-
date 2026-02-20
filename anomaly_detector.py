import numpy as np

# Fusion imports
from fusion.score_normalizer import normalize_scores
from fusion.fusion_logic import fuse_modalities
from fusion.threshold import compute_threshold, predict_anomalies

# Evaluation imports
from evaluation.metrics import compute_classification_metrics
from evaluation.visualization import plot_roc_curve


class AnomalyDetector:
    """
    End-to-end anomaly detector:
    Sensor + Image + Log → Fusion → Threshold → Prediction → Evaluation
    """

    def __init__(
        self,
        fusion_method="weighted",
        fusion_weights=(0.4, 0.4, 0.2),
        normalization="zscore",
        threshold_method="percentile",
        threshold_value=95,
    ):
        self.fusion_method = fusion_method
        self.fusion_weights = fusion_weights
        self.normalization = normalization
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value

  
    # Step 1: Normalize modality scores
   
    def normalize(self, sensor_scores, image_scores, log_scores):
        sensor_n = normalize_scores(sensor_scores, self.normalization)
        image_n = normalize_scores(image_scores, self.normalization)
        log_n = normalize_scores(log_scores, self.normalization)
        return sensor_n, image_n, log_n

    
    # Step 2: Fuse normalized scores
  
    def fuse(self, sensor_n, image_n, log_n):
        fused_scores = fuse_modalities(
            sensor_n,
            image_n,
            log_n,
            method=self.fusion_method,
            weights=self.fusion_weights,
        )
        return fused_scores

   
    # Step 3: Thresholding
   
    def threshold(self, fused_scores):
        threshold = compute_threshold(
            fused_scores,
            method=self.threshold_method,
            value=self.threshold_value,
        )
        return threshold

    # Step 4: Predict anomalies
   
    def predict(self, fused_scores, threshold):
        predictions = predict_anomalies(fused_scores, threshold)
        return predictions

   
    # Step 5: Full inference pipeline
    
    def run(self, sensor_scores, image_scores, log_scores, y_true=None):
        sensor_n, image_n, log_n = self.normalize(
            sensor_scores, image_scores, log_scores
        )

        fused_scores = self.fuse(sensor_n, image_n, log_n)

        threshold = self.threshold(fused_scores)

        predictions = self.predict(fused_scores, threshold)

        results = {
            "fused_scores": fused_scores,
            "threshold": threshold,
            "predictions": predictions,
        }

        # Optional evaluation
        if y_true is not None:
            metrics = compute_classification_metrics(y_true, predictions)
            plot_roc_curve(y_true, fused_scores)
            results["metrics"] = metrics

        return results


# Run example (for testing)

if __name__ == "__main__":
    print("Running anomaly detector...")

    # Dummy scores (replace with real model outputs)
    sensor_scores = np.random.rand(100)
    image_scores = np.random.rand(100)
    log_scores = np.random.rand(100)

    # Ground truth (optional)
    y_true = np.random.randint(0, 2, size=100)

    detector = AnomalyDetector()

    outputs = detector.run(
        sensor_scores=sensor_scores,
        image_scores=image_scores,
        log_scores=log_scores,
        y_true=y_true,
    )

    print("Threshold:", outputs["threshold"])
    print("Sample predictions:", outputs["predictions"][:10])
    print("Metrics:", outputs.get("metrics"))
