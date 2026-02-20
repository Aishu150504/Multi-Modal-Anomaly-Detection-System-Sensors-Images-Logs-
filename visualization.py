import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    print("Running visualization module...")

    # Dummy test data
    y_true = [0, 0, 1, 1, 1, 0]
    y_scores = [0.1, 0.3, 0.8, 0.9, 0.7, 0.2]

    plot_roc_curve(y_true, y_scores)
