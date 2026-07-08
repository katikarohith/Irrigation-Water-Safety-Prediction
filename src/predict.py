"""
predict.py
----------
Loads the saved model + scaler and produces irrigation safety predictions
(with confidence scores) for new water-quality samples. Used by both the
CLI demo below and the Streamlit app.

CLI usage:
    python src/predict.py
"""

import logging
import os
from typing import Dict, Union

import joblib
import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_model = None
_scaler = None


def _load_artifacts():
    """Lazily load and cache the model and scaler so repeated calls are cheap."""
    global _model, _scaler
    if _model is None or _scaler is None:
        if not os.path.exists(config.BEST_MODEL_PATH) or not os.path.exists(config.SCALER_PATH):
            raise FileNotFoundError(
                "Trained model/scaler not found. Run `python src/train_model.py` first."
            )
        _model = joblib.load(config.BEST_MODEL_PATH)
        _scaler = joblib.load(config.SCALER_PATH)
        logger.info("Loaded model and scaler from disk.")
    return _model, _scaler


def validate_input(sample: Dict[str, float]) -> None:
    """
    Validate that a raw input dict contains all required features with
    numeric, non-negative-where-applicable values.

    Raises
    ------
    ValueError
        If any required feature is missing or has an invalid type/value.
    """
    missing = [f for f in config.FEATURE_COLUMNS if f not in sample]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    for feature in config.FEATURE_COLUMNS:
        value = sample[feature]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Feature '{feature}' must be numeric, got {type(value).__name__}")
        if np.isnan(value):
            raise ValueError(f"Feature '{feature}' cannot be NaN")

    if not (0 <= sample["ph"] <= 14):
        raise ValueError(f"'ph' must be between 0 and 14, got {sample['ph']}")


def predict_single(sample: Dict[str, float]) -> Dict[str, Union[str, float]]:
    """
    Predict irrigation safety for a single water sample.

    Parameters
    ----------
    sample : dict
        Must contain keys matching config.FEATURE_COLUMNS
        (ph, Hardness, Solids, Chloramines, Sulfate, Conductivity,
        Organic_carbon, Trihalomethanes, Turbidity).

    Returns
    -------
    dict with keys: 'prediction' (str), 'prediction_code' (int),
    'confidence' (float, 0-1).
    """
    validate_input(sample)
    model, scaler = _load_artifacts()

    input_df = pd.DataFrame([sample])[config.FEATURE_COLUMNS]
    input_scaled = scaler.transform(input_df)

    prediction_code = int(model.predict(input_scaled)[0])

    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(input_scaled)[0][prediction_code])
    else:
        confidence = 1.0  # model has no probability estimate available

    return {
        "prediction": config.LABEL_MAP[prediction_code],
        "prediction_code": prediction_code,
        "confidence": round(confidence, 4),
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict irrigation safety for multiple samples at once.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all columns in config.FEATURE_COLUMNS.

    Returns
    -------
    pd.DataFrame
        Original dataframe with two new columns: 'Prediction', 'Confidence'.
    """
    missing = set(config.FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Input dataframe is missing required column(s): {missing}")

    model, scaler = _load_artifacts()

    X = df[config.FEATURE_COLUMNS]
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    result_df = df.copy()
    result_df["Prediction"] = [config.LABEL_MAP[p] for p in preds]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)
        result_df["Confidence"] = [round(proba[i][preds[i]], 4) for i in range(len(preds))]
    else:
        result_df["Confidence"] = 1.0

    return result_df


if __name__ == "__main__":
    # Sample prediction demo (values within safe irrigation thresholds)
    sample_safe = {
        "ph": 7.2,
        "Hardness": 200.0,
        "Solids": 15000.0,
        "Chloramines": 6.5,
        "Sulfate": 300.0,
        "Conductivity": 450.0,
        "Organic_carbon": 12.0,
        "Trihalomethanes": 60.0,
        "Turbidity": 3.5,
    }

    sample_unsafe = {
        "ph": 5.1,
        "Hardness": 180.0,
        "Solids": 32000.0,
        "Chloramines": 7.8,
        "Sulfate": 280.0,
        "Conductivity": 3400.0,
        "Organic_carbon": 15.0,
        "Trihalomethanes": 70.0,
        "Turbidity": 4.1,
    }

    for label, sample in [("Sample A (expected SAFE)", sample_safe),
                           ("Sample B (expected UNSAFE)", sample_unsafe)]:
        result = predict_single(sample)
        print(f"\n{label}")
        print(f"  Prediction : {result['prediction']}")
        print(f"  Confidence : {result['confidence'] * 100:.2f}%")
