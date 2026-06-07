import os
import argparse

import pandas as pd
from joblib import load

from .logging_utils import get_logger
from . import data_utils, featurization
from .utils import positive_proba

logger = get_logger("triage")


def main():
    ap = argparse.ArgumentParser(description="Hit triage ranking on test split")
    ap.add_argument("--data", required=True, help="CSV con colonne 'smiles','activity'")
    ap.add_argument("--rep", required=True, choices=["morgan", "maccs"])
    ap.add_argument("--model-pkl", required=True, help="Percorso al modello .pkl")
    ap.add_argument("--outdir", required=True, help="Directory output")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--morgan-radius", type=int, default=2)
    ap.add_argument("--morgan-bits", type=int, default=2048)
    ap.add_argument("--topn", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # load dati + split
    df = data_utils.load_data(args.data)
    df_train, df_test = data_utils.train_test_split(
        df, test_size=args.test_size, seed=args.seed
    )

    # featurization SOLO sul test
    if args.rep == "morgan":
        X_test = featurization.build_features(
            df_test, rep="morgan", radius=args.morgan_radius, n_bits=args.morgan_bits
        )
    else:
        X_test = featurization.build_features(df_test, rep="maccs")

    # carica modello già addestrato
    clf = load(args.model_pkl)

    # probabilità di essere attivo
    y_prob = positive_proba(clf, X_test, pos_label=1)

    # ranking
    out = df_test.copy()
    out["prob_active"] = y_prob
    out = out.sort_values("prob_active", ascending=False)
    topn = out.head(args.topn)

    out_path = os.path.join(args.outdir, f"triage_top{args.topn}.csv")
    topn.to_csv(out_path, index=False)

    logger.info("Saved hit triage list at %s", out_path)


if __name__ == "__main__":
    main()