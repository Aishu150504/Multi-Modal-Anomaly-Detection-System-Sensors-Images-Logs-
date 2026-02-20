import numpy as np

def compute_threshold(scores, method="percentile", value=95):
    scores = np.array(scores)

    if method == "percentile":
        return np.percentile(scores, value)
    elif method == "mean_std":
        return scores.mean() + 3 * scores.std()
    else:
        raise ValueError("Invalid threshold method")

def predict_anomalies(scores, threshold):
    scores = np.array(scores)
    return (scores > threshold).astype(int)
