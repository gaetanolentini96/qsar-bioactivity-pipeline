import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from .logging_utils import get_logger

logger = get_logger(__name__)


def train_rf(X, y, seed: int = 42):
    """
    Train a Random Forest classifier.
    - Uses class_weight='balanced'
    - Applies isotonic calibration (cv=5) only if there are enough samples per class.
    """
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )

    y = np.asarray(y)
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    min_count = counts.min()

    # Usa calibrazione solo se ogni classe ha almeno 5 campioni
    if min_count >= 5:
        calib = CalibratedClassifierCV(rf, method="isotonic", cv=5)
        calib.fit(X, y)
        logger.info(
            "Trained RandomForest + isotonic calibration (cv=5) | n_samples=%d | n_features=%d | class_counts=%s",
            X.shape[0],
            X.shape[1],
            class_counts,
        )
        return calib

    # Dataset piccolo → niente calibrazione, per evitare errori
    logger.warning(
        "Too few samples per class for calibration (min_count=%d). "
        "Training plain RandomForest without CalibratedClassifierCV.",
        min_count,
    )

    rf.fit(X, y)
    logger.info(
        "Trained RandomForest (no calibration) | n_samples=%d | n_features=%d | class_counts=%s",
        X.shape[0],
        X.shape[1],
        class_counts,
    )
    return rf


def train_xgb(X, y, seed: int = 42):
    """
    Train an XGBoost classifier with reasonable defaults.
    """
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        raise RuntimeError(
            "XGBoost not installed. Please install `xgboost` or use RF."
        ) from e

    y = np.asarray(y)
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))

    xgb = XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        tree_method="hist",
        n_jobs=-1,
        eval_metric="logloss",
        scale_pos_weight=None,
    )

    xgb.fit(X, y)

    logger.info(
        "Trained XGBoost | n_samples=%d | n_features=%d | class_counts=%s",
        X.shape[0],
        X.shape[1],
        class_counts,
    )

    return xgb