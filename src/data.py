"""Load and audit the original CC BY 4.0 Iranian Churn dataset from UCI."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

DATASET_NAME = "Iranian Churn (UCI ID 563)"
SOURCE_URL = "https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset"
DOI = "https://doi.org/10.24432/C5JW3Z"
EXPECTED_COLUMNS = (
    "Call Failure", "Complains", "Subscription Length", "Charge Amount",
    "Seconds of Use", "Frequency of use", "Frequency of SMS",
    "Distinct Called Numbers", "Age Group", "Tariff Plan", "Status", "Age",
    "Customer Value", "Churn",
)
TARGET = "Churn"
# Intentionally conservative predictors: status may be close to the outcome;
# customer value has an unspecified derivation; age is excluded from prediction.
NUMERIC_FEATURES = (
    "Call Failure", "Subscription Length", "Charge Amount",
    "Seconds of Use", "Frequency of use", "Frequency of SMS",
    "Distinct Called Numbers",
)
CATEGORICAL_FEATURES = ("Complains", "Tariff Plan")
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
EXCLUDED_FEATURES = ("Status", "Customer Value", "Age", "Age Group")


def load_uci_csv(path: str | Path, *, strict_rows: bool = True) -> pd.DataFrame:
    """Load only the expected 14-column UCI schema; reject partial/incorrect data.

    The official repository describes 3,150 customers. A special relaxed mode is
    reserved for unit tests using *synthetic in-memory fixtures*.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run `python scripts/download_data.py` "
            "from the project root while online."
        )
    df = pd.read_csv(path)
    # UCI CSV headers can contain non-breaking spaces (U+00A0), which
    # str.strip() does not remove when they occur between words.
    # Normalize all Unicode whitespace runs without changing data values.
    df.columns = [" ".join(str(column).split()) for column in df.columns]
    if len(df.columns) != len(set(df.columns)):
        raise ValueError("Duplicate CSV column names")
    if set(df.columns) != set(EXPECTED_COLUMNS):
        missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
        unexpected = sorted(set(df.columns) - set(EXPECTED_COLUMNS))
        raise ValueError(f"Unexpected CSV schema. Missing: {missing}; extra: {unexpected}")
    df = df.loc[:, list(EXPECTED_COLUMNS)].copy()
    if strict_rows and len(df) != 3150:
        raise ValueError(f"Expected 3,150 UCI records; found {len(df)}")
    if df.empty or df.isna().any().any():
        raise ValueError("Dataset is empty or contains missing entries")
    for col in EXPECTED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")
    for col in EXPECTED_COLUMNS:
        if not df[col].map(lambda x: pd.notna(x) and float('-inf') < x < float('inf')).all():
            raise ValueError(f"Non-finite values in {col}")
    if not set(df[TARGET].unique()).issubset({0, 1}) or df[TARGET].nunique() != 2:
        raise ValueError("Churn must contain both 0 and 1 classes")
    for col, allowed in {"Complains": {0, 1}, "Tariff Plan": {1, 2},
                         "Status": {1, 2}, "Age Group": {1, 2, 3, 4, 5}}.items():
        if not set(df[col].unique()).issubset(allowed):
            raise ValueError(f"Unexpected category in {col}")
    if (df[list(NUMERIC_FEATURES)] < 0).any().any():
        raise ValueError("Nonnegative operational feature has negative values")
    return df


def data_audit(df: pd.DataFrame) -> dict:
    """Useful counts that cannot be inferred from metadata alone."""
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "positive_count": int(df[TARGET].sum()),
        "prevalence": float(df[TARGET].mean()),
        "missing_values": int(df.isna().sum().sum()),
        "identical_full_rows": int(df.duplicated().sum()),
        "duplicate_feature_vectors": int(df[list(MODEL_FEATURES)].duplicated().sum()),
        "conflicting_target_profiles": int((
            df.groupby(list(MODEL_FEATURES), dropna=False)[TARGET].nunique() > 1
        ).sum()),
    }
