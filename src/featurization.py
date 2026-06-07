from typing import List

import numpy as np
import pandas as pd

from .logging_utils import get_logger

logger = get_logger(__name__)


def _require_rdkit():
    try:
        from rdkit import Chem  # noqa: F401
        from rdkit.Chem import AllChem, MACCSkeys  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "RDKit is required for featurization. Install via conda or rdkit-pypi."
        ) from e


def morgan_fp(smiles_list: List[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    _require_rdkit()
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

    fps = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            logger.warning("Invalid SMILES for Morgan FP: %s", s)
            fps.append(np.zeros(n_bits, dtype=np.uint8))
            continue

        bv = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        ConvertToNumpyArray(bv, arr)
        fps.append(arr)

    return np.array(fps)


def maccs_fp(smiles_list: List[str]) -> np.ndarray:
    _require_rdkit()
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

    fps = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            logger.warning("Invalid SMILES for MACCS FP: %s", s)
            fps.append(np.zeros(167, dtype=np.uint8))
            continue

        bv = MACCSkeys.GenMACCSKeys(m)
        arr = np.zeros((167,), dtype=np.uint8)
        ConvertToNumpyArray(bv, arr)
        fps.append(arr)

    return np.array(fps)


def build_features(df: pd.DataFrame, rep: str = "morgan", **kwargs) -> np.ndarray:
    smiles = df["smiles"].tolist()

    if rep == "morgan":
        X = morgan_fp(smiles, **kwargs)
    elif rep == "maccs":
        X = maccs_fp(smiles)
    else:
        raise ValueError(f"Unknown representation: {rep}")

    logger.info("Built features: rep=%s | shape=%s", rep, X.shape)
    return X