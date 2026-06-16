import argparse
import os
import json
import numpy as np
from joblib import dump

from .logging_utils import get_logger
from . import data_utils, featurization, model as mdl, eval as evl
from .plotting import plot_roc, plot_pr, plot_calibration, plot_confusion_matrix
from .utils import positive_proba

logger = get_logger("pipeline")


def main():
    ap = argparse.ArgumentParser(description="Bioactivity ML Pipeline (single run)")
    ap.add_argument("--data", required=True, help="Path to CSV with 'smiles' and 'activity' columns")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--rep", default="morgan", choices=["morgan", "maccs"], help="Molecular representation")
    ap.add_argument("--model", default="rf", choices=["rf", "xgb"], help="Model type")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--morgan-radius", type=int, default=2)
    ap.add_argument("--morgan-bits", type=int, default=2048)
    args = ap.parse_args()

    # --- setup cartelle ---
    os.makedirs(args.outdir, exist_ok=True)
    features_dir = os.path.join(args.outdir, "features")
    reports_dir = os.path.join(args.outdir, "reports")
    models_dir = os.path.join(args.outdir, "models")
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # --- load & split ---
    df = data_utils.load_data(args.data)
    df_train, df_test = data_utils.train_test_split(df, test_size=args.test_size, seed=args.seed)

    # --- featurization ---
    if args.rep == "morgan":
        X_train = featurization.build_features(df_train, rep="morgan", radius=args.morgan_radius, n_bits=args.morgan_bits)
        X_test = featurization.build_features(df_test, rep="morgan", radius=args.morgan_radius, n_bits=args.morgan_bits)
    else:
        X_train = featurization.build_features(df_train, rep="maccs")
        X_test = featurization.build_features(df_test, rep="maccs")

    y_train = df_train["activity"].values
    y_test = df_test["activity"].values

    # --- training ---
    if args.model == "rf":
        clf = mdl.train_rf(X_train, y_train, seed=args.seed)
    else:
        clf = mdl.train_xgb(X_train, y_train, seed=args.seed)

    # --- evaluation ---
    y_prob = positive_proba(clf, X_test, pos_label=1)
    metrics = evl.classification_metrics(y_test, y_prob, threshold=0.15)
    (fpr, tpr, _), (prec, rec, _) = evl.curves(y_test, y_prob)
    prob_true, prob_pred = evl.calibration_points(y_test, y_prob, n_bins=10)

    # --- save plots ---
    plot_roc(fpr, tpr, os.path.join(reports_dir, "roc.png"))
    plot_pr(rec, prec, os.path.join(reports_dir, "pr.png"))
    plot_calibration(prob_true, prob_pred, os.path.join(reports_dir, "calibration.png"))
    tn, fp, fn, tp = metrics["tn_fp_fn_tp"]
    plot_confusion_matrix(tn, fp, fn, tp, os.path.join(reports_dir, "confusion.png"))

    # --- salva risultati ---
    dump(clf, os.path.join(models_dir, f"{args.model}_{args.rep}.pkl"))
    np.savez_compressed(os.path.join(features_dir, "X_train.npz"), X_train=X_train, y_train=y_train)
    np.savez_compressed(os.path.join(features_dir, "X_test.npz"), X_test=X_test, y_test=y_test)
    with open(os.path.join(reports_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        "Done. ROC-AUC=%.3f PR-AUC=%.3f | Brier=%.3f",
        metrics["roc_auc"], metrics["pr_auc"], metrics["brier"]
    )


if __name__ == "__main__":
    main()