import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def compute_classification_metrics(y_true, y_pred):
    """
    Compute basic classification metrics
    """
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }

def compute_auc(y_true, anomaly_scores):
    """
    Compute ROC-AUC using anomaly scores
    """
    return roc_auc_score(y_true, anomaly_scores)
