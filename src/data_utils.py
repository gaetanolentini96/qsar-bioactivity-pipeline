import pandas as pd
from .logging_utils import get_logger  # nota il punto: stesso package

logger = get_logger(__name__)


def load_data(path: str) -> pd.DataFrame:
    """
    CSV richiesto con colonne:
    - smiles (str)
    - activity (0/1)
    """
    df = pd.read_csv(path)

    if "smiles" not in df.columns or "activity" not in df.columns:
        raise ValueError("CSV must contain 'smiles' and 'activity' columns")

    df = df.dropna(subset=["smiles", "activity"]).copy()
    df["activity"] = df["activity"].astype(int)

    before = len(df)
    df = df.drop_duplicates(subset=["smiles"])
    logger.info("Dropped %d duplicate rows", before - len(df))

    return df


def train_test_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    from sklearn.model_selection import train_test_split as sk_split

    return sk_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["activity"]
    )