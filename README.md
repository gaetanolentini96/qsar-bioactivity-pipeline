<<<<<<< HEAD
# QSAR Bioactivity Pipeline (ECFP vs MACCS, RF vs XGBoost)

End-to-end QSAR pipeline for binary bioactivity prediction, built in Python with RDKit and scikit-learn.  
It compares different **molecular representations** and **models**, handles **class imbalance**, and demonstrates a realistic **hit triage** use case.

This project was designed as an *industry-style* chemoinformatics portfolio project.

---

## 🔍 Features

- **Data handling**
  - Load CSV assay data with `smiles` and `activity` (0/1)
  - Basic cleaning: `dropna`, cast `activity` to `int`, remove duplicate SMILES

- **Featurization (RDKit)**
  - ECFP-like **Morgan fingerprints** (`radius`, `n_bits` configurable)
  - **MACCS** keys (167 bits)
  - Graceful handling of invalid SMILES

- **Models**
  - **Random Forest** (class_weight='balanced')
  - **XGBoost** (tree_method="hist", reasonable defaults for imbalanced data)
  - Optional calibration (isotonic) when enough samples are available

- **Evaluation**
  - Metrics:
    - ROC-AUC
    - PR-AUC (more informative for imbalanced datasets)
    - Brier score
    - Confusion matrix (@ threshold 0.5)
  - Curves:
    - ROC curve
    - Precision–Recall curve
    - Calibration curve

- **Experiments (PRO mode)**
  - Grid over:
    - Representations: `morgan`, `maccs`
    - Models: `rf`, `xgb`
  - Per-run outputs:
    - `metrics.json`
    - `roc.png`, `pr.png`, `calibration.png`, `confusion.png`
  - Global outputs:
    - `experiments_summary.csv`
    - `best_run.json` (selected by **PR-AUC**)

- **Hit triage**
  - Given a trained model, rank test-set molecules by predicted probability of being active
  - Export top-N compounds as `triage_topN.csv`

---

## 📂 Project structure

```text
your-repo/
├── data/
│   └── toy_assay.csv              # small demo dataset (SMILES + activity)
├── outputs_test/                  # example outputs from pipeline.py
├── outputs_experiments/           # example outputs from experiments.py
├── outputs_triage/                # example outputs from triage.py
├── src/
│   ├── __init__.py
│   ├── logging_utils.py
│   ├── data_utils.py
│   ├── featurization.py
│   ├── model.py
│   ├── eval.py
│   ├── plotting.py
│   ├── utils.py                   # positive_proba helper
│   ├── pipeline.py                # single run (rep + model)
│   ├── experiments.py             # PRO: models x representations
│   └── triage.py                  # hit triage ranking (top-N)
├── requirements.txt
└── README.md
=======
# qsar-bioactivity-pipeline
Cheminformatics QSAR workflow for bioactivity prediction with RDKit, machine learning, and compound prioritization.
>>>>>>> b2ea66e81e0203fb8e3c5b5804f4d784e8dacc61
