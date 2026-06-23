import matplotlib.pyplot as plt
import numpy as np


def plot_roc(fpr, tpr, out_png: str):
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()


def plot_pr(rec, prec, out_png: str):
    plt.figure()
    plt.plot(rec, prec, label="Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()


def plot_calibration(prob_true, prob_pred, out_png: str):
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Curve")
    plt.legend()
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()


def plot_confusion_matrix(tn, fp, fn, tp, out_png: str):
    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure()
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title("Confusion Matrix @0.15")
    plt.colorbar()
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()