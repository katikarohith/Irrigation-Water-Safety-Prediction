"""
data_preprocessing.py
----------------------
Handles loading of the raw water-quality dataset, derivation of the
Irrigation_Safety target label, missing-value imputation, and the
train/test split + feature scaling used by every downstream script.

Run directly to produce a cleaned copy of the dataset in
data/processed/processed_data.csv:

    python src/data_preprocessing.py
"""

import os
import logging

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_data(path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw water quality CSV into a DataFrame.

    Parameters
    ----------
    path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    ValueError
        If the file is empty or cannot be parsed.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw data file not found at '{path}'. "
            "Place 'water_potability.csv' inside the data/raw/ folder."
        )
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"The file at '{path}' is empty.") from exc
    except Exception as exc:  # noqa: BLE001 - surface a clear, actionable error
        raise ValueError(f"Failed to read CSV at '{path}': {exc}") from exc

    if df.empty:
        raise ValueError("Loaded dataframe is empty. Check the source CSV.")

    logger.info("Loaded raw data: %d rows, %d columns", *df.shape)
    return df


def derive_irrigation_safety_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the binary 'Irrigation_Safety' target column using domain
    thresholds for electrical conductivity, total dissolved solids (TDS)
    and pH — the three parameters most commonly used by agronomists to
    judge whether water is suitable for crop irrigation.

    Water is labeled:
        1 -> Safe for irrigation      (all three conditions satisfied)
        0 -> Unsafe for irrigation    (any one condition violated)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Conductivity', 'Solids', and 'ph' columns.

    Returns
    -------
    pd.DataFrame
        Copy of the input dataframe with the new target column appended.
    """
    required = {"Conductivity", "Solids", "ph"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Cannot derive target label - missing columns: {missing}")

    df = df.copy()

    def _label_row(row: pd.Series) -> int:
        conductivity_ok = row["Conductivity"] < config.CONDUCTIVITY_MAX
        tds_ok = row["Solids"] < config.TDS_MAX
        ph_ok = config.PH_MIN <= row["ph"] <= config.PH_MAX
        return int(conductivity_ok and tds_ok and ph_ok)

    df[config.TARGET_COLUMN] = df.apply(_label_row, axis=1)
    logger.info(
        "Derived target label. Class balance -> %s",
        df[config.TARGET_COLUMN].value_counts().to_dict(),
    )
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using the strategy defined in config.IMPUTATION_STRATEGY
    (mean for 'ph', median for 'Sulfate' and 'Trihalomethanes' — matching
    the original exploratory analysis, which found these columns skewed).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Dataframe with no remaining missing values in the configured columns.
    """
    df = df.copy()
    for column, strategy in config.IMPUTATION_STRATEGY.items():
        if column not in df.columns:
            logger.warning("Column '%s' not found; skipping imputation.", column)
            continue

        if strategy == "mean":
            fill_value = df[column].mean()
        elif strategy == "median":
            fill_value = df[column].median()
        else:
            raise ValueError(f"Unsupported imputation strategy '{strategy}' for '{column}'")

        n_missing = df[column].isnull().sum()
        df[column] = df[column].fillna(fill_value)
        logger.info("Filled %d missing values in '%s' using %s (%.3f)",
                    n_missing, column, strategy, fill_value)

    remaining_na = df[list(config.IMPUTATION_STRATEGY.keys())].isnull().sum().sum()
    if remaining_na > 0:
        logger.warning("There are still %d missing values after imputation.", remaining_na)

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows, logging how many were removed."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        logger.info("Removed %d duplicate rows.", removed)
    return df


def split_and_scale(df: pd.DataFrame):
    """
    Split the dataframe into train/test sets and standardize the features.

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test, scaler : tuple
    """
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Train shape: %s | Test shape: %s", X_train.shape, X_test.shape)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def run_preprocessing_pipeline(save: bool = True) -> pd.DataFrame:
    """
    Full preprocessing pipeline: load -> derive label -> clean -> dedupe.
    Optionally saves the cleaned dataset to data/processed/.

    Returns
    -------
    pd.DataFrame
        The fully cleaned dataframe (still unscaled/unsplit).
    """
    df = load_data()
    df = derive_irrigation_safety_label(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)

    if save:
        os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
        df.to_csv(config.PROCESSED_DATA_PATH, index=False)
        logger.info("Saved processed data to %s", config.PROCESSED_DATA_PATH)

    return df


if __name__ == "__main__":
    run_preprocessing_pipeline()
