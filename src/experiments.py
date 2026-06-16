import os
import json
import argparse
from itertools import product

import pandas as pd

from .logging_utils import get_logger
from . import data_utils, featurization, model as mdl, eval as evl
from .plotting import plot_roc, plot_pr, plot_calibration, plot_confusion_matrix
from .utils import positive_proba

logger = get_logger("experiments")


def run_single(
    df: pd.DataFrame,
    outdir: str,
    rep: str,
    model: str,
    seed: int = 42,
    test_size: float = 0.2,
    radius: int = 2,
    n_bits: int = 2048,
):
    """
    Esegue un singolo esperimento (una rappresentazione + un modello)
    e salva tutti i risultati in una sottocartella: outdir/model_rep/
    """
    # --- prepara cartelle ---
    run_dir = os.path.join(outdir, f"{model}_{rep}")
    reports_dir = os.path.join(run_dir, "reports")
    models_dir = os.path.join(run_dir, "models")
    feats_dir = os.path.join(run_dir, "features")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(feats_dir, exist_ok=True)

    # --- train/test split ---
    df_train, df_test = data_utils.train_test_split(df, test_size=test_size, seed=seed)

    # --- featurization ---
    if rep == "morgan":
        X_train = featurization.build_features(
            df_train, rep="morgan", radius=radius, n_bits=n_bits
        )
        X_test = featurization.build_features(
            df_test, rep="morgan", radius=radius, n_bits=n_bits
        )
    else:  # maccs
        X_train = featurization.build_features(df_train, rep="maccs")
        X_test = featurization.build_features(df_test, rep="maccs")

    y_train = df_train["activity"].values
    y_test = df_test["activity"].values

    # --- training ---
    if model == "rf":
        clf = mdl.train_rf(X_train, y_train, seed=seed)
    else:  # xgb
        clf = mdl.train_xgb(X_train, y_train, seed=seed)

    # --- evaluation ---
    y_prob = positive_proba(clf, X_test, pos_label=1)
    metrics = evl.classification_metrics(y_test, y_prob, threshold=0.15)
    (fpr, tpr, _), (prec, rec, _) = evl.curves(y_test, y_prob)
    prob_true, prob_pred = evl.calibration_points(y_test, y_prob)

    # --- plots ---
    plot_roc(fpr, tpr, os.path.join(reports_dir, "roc.png"))
    plot_pr(rec, prec, os.path.join(reports_dir, "pr.png"))
    plot_calibration(prob_true, prob_pred, os.path.join(reports_dir, "calibration.png"))
    tn, fp, fn, tp = metrics["tn_fp_fn_tp"]
    plot_confusion_matrix(
        tn, fp, fn, tp, os.path.join(reports_dir, "confusion.png")
    )

    # --- salvataggio risultati singolo run ---
    # metrics.json include il tipo di modello e la rappresentazione
    with open(os.path.join(reports_dir, "metrics.json"), "w") as f:
        json.dump({"rep": rep, "model": model, **metrics}, f, indent=2)

    logger.info(
        "Run %s_%s: PR-AUC=%.3f | ROC-AUC=%.3f",
        model,
        rep,
        metrics["pr_auc"],
        metrics["roc_auc"],
    )

    return run_dir, metrics


def main():
    ap = argparse.ArgumentParser(
        description="Run PRO experiments: models x representations"
    )
    ap.add_argument("--data", required=True, help="Path CSV con 'smiles','activity'")
    ap.add_argument("--outdir", required=True, help="Directory output per gli esperimenti")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--morgan-radius", type=int, default=2)
    ap.add_argument("--morgan-bits", type=int, default=2048)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = data_utils.load_data(args.data)

    rows = []

    # tutte le combinazioni: (morgan/maccs) x (rf/xgb)
    for rep, model in product(["morgan", "maccs"], ["rf", "xgb"]):
        run_dir, metrics = run_single(
            df=df,
            outdir=args.outdir,
            rep=rep,
            model=model,
            seed=args.seed,
            test_size=args.test_size,
            radius=args.morgan_radius,
            n_bits=args.morgan_bits,
        )
        rows.append({"run_dir": run_dir, "rep": rep, "model": model, **metrics})

    # --- summary tabellare ---
    summary = pd.DataFrame(rows)
    summary_path = os.path.join(args.outdir, "experiments_summary.csv")
    summary.to_csv(summary_path, index=False)

    # --- best run by PR-AUC (più adatta per classi sbilanciate) ---
    best_row = summary.sort_values("pr_auc", ascending=False).iloc[0].to_dict()
    with open(os.path.join(args.outdir, "best_run.json"), "w") as f:
        json.dump(best_row, f, indent=2)

    logger.info(
        "Best run by PR-AUC: %s_%s | PR-AUC=%.3f | ROC-AUC=%.3f",
        best_row["model"],
        best_row["rep"],
        best_row["pr_auc"],
        best_row["roc_auc"],
    )
    logger.info("Saved summary at %s", summary_path)


if __name__ == "__main__":
    main()