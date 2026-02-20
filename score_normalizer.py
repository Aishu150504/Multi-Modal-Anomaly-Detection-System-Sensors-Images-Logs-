import numpy as np

def normalize_scores(scores, method="minmax"):
    """
    Normalize anomaly scores to [0, 1]

    Args:
        scores (list or np.ndarray): raw anomaly scores
        method (str): 'minmax' or 'zscore'

    Returns:
        np.ndarray: normalized scores
    """
    scores = np.array(scores, dtype=float)

    if method == "minmax":
        min_s = scores.min()
        max_s = scores.max()
        if max_s - min_s == 0:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    elif method == "zscore":
        mean = scores.mean()
        std = scores.std()
        if std == 0:
            return np.zeros_like(scores)
        return (scores - mean) / std

    else:
        raise ValueError("method must be 'minmax' or 'zscore'")
