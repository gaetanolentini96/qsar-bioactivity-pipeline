from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

from .logging_utils import get_logger

logger = get_logger(__name__)


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calcola metriche principali per classificazione binaria.
    Restituisce:
    - roc_auc
    - pr_auc
    - brier
    - soglia usata
    - tn, fp, fn, tp
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if y_true.ndim != 1:
        raise ValueError("y_true must be 1D")
    if y_prob.ndim != 1:
        raise ValueError("y_prob must be 1D")
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have same length")

    # Predizioni binarie
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Metriche robuste (proteggiamo da errori su dataset troppo piccoli)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc = float("nan")

    try:
        pr = average_precision_score(y_true, y_prob)
    except ValueError:
        pr = float("nan")

    try:
        brier = brier_score_loss(y_true, y_prob)
    except ValueError:
        brier = float("nan")

    metrics = {
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "brier": float(brier),
        "threshold": float(threshold),
        "tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
    }

    logger.info("Metrics: %s", metrics)
    return metrics


def curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Ritorna punti per ROC e Precision-Recall.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
    prec, rec, pr_thr = precision_recall_curve(y_true, y_prob)

    return (fpr, tpr, roc_thr), (prec, rec, pr_thr)


def calibration_points(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
):
    """
    Punti per curva di calibrazione (pred vs osservato).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy="quantile",
    )

    return prob_true, prob_pred