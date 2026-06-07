import numpy as np

def positive_proba(clf, X, pos_label=1):
    """Restituisce sempre P(y==pos_label)."""
    if not hasattr(clf, "classes_"):
        raise AttributeError("Classifier has no attribute 'classes_'. Was it fitted?")
    classes = np.array(clf.classes_)
    idx = np.where(classes == pos_label)[0]
    if len(idx) == 0:
        raise ValueError(f"pos_label {pos_label} not in clf.classes_={classes}")
    return clf.predict_proba(X)[:, int(idx[0])]