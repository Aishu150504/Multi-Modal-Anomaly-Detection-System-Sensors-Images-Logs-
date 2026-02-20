import numpy as np

def fuse_modalities(
    sensor_scores,
    image_scores,
    log_scores,
    method="weighted",
    weights=(0.4, 0.4, 0.2)
):
    sensor_scores = np.array(sensor_scores)
    image_scores = np.array(image_scores)
    log_scores = np.array(log_scores)

    if method == "weighted":
        w_s, w_i, w_l = weights
        return w_s * sensor_scores + w_i * image_scores + w_l * log_scores

    elif method == "mean":
        return (sensor_scores + image_scores + log_scores) / 3.0

    elif method == "max":
        return np.maximum.reduce([sensor_scores, image_scores, log_scores])

    else:
        raise ValueError(f"Unknown fusion method: {method}")


