"""
train_model.py
---------------
Trains Logistic Regression and Random Forest classifiers to predict
irrigation water safety, tunes each with RandomizedSearchCV, selects the
best-performing model based on test accuracy, and persists the winning
model + scaler to the models/ folder using joblib.

Run directly:
    python src/train_model.py
"""

import logging
import os
import warnings

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV

import config
from data_preprocessing import run_preprocessing_pipeline, split_and_scale

# Silence noisy (non-actionable) sklearn deprecation/parameter warnings that
# occur naturally during RandomizedSearchCV's broad hyperparameter sweep.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Candidate models
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(random_state=config.RANDOM_STATE),
}

# Hyperparameter search spaces (mirrors the original notebook)
PARAM_GRID = {
    "Logistic Regression": {
        "penalty": ["l1", "l2"],
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["liblinear", "saga"],
        "max_iter": [200, 500],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300, 500],
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True, False],
    },
}


def evaluate(model, X, y) -> dict:
    """Compute standard classification metrics for a fitted model."""
    y_pred = model.predict(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
    }


def tune_models(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """
    Run RandomizedSearchCV for every model in MODELS, evaluate the best
    estimator of each on the held-out test set, and return a results table.

    Returns
    -------
    pd.DataFrame with columns: Model, estimator, Best Params, Best CV Score,
    Train metrics, Test metrics.
    """
    records = []

    for name, model in MODELS.items():
        logger.info("Tuning %s ...", name)
        try:
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=PARAM_GRID[name],
                n_iter=20,
                scoring="accuracy",
                cv=3,
                n_jobs=-1,
                verbose=0,
                random_state=config.RANDOM_STATE,
            )
            search.fit(X_train, y_train)
        except Exception as exc:  # noqa: BLE001
            logger.error("Hyperparameter tuning failed for %s: %s", name, exc)
            continue

        best_estimator = search.best_estimator_
        train_metrics = evaluate(best_estimator, X_train, y_train)
        test_metrics = evaluate(best_estimator, X_test, y_test)

        records.append({
            "Model": name,
            "estimator": best_estimator,
            "Best Params": search.best_params_,
            "Best CV Score": round(search.best_score_, 4),
            "Train Accuracy": round(train_metrics["Accuracy"], 4),
            "Train Precision": round(train_metrics["Precision"], 4),
            "Train Recall": round(train_metrics["Recall"], 4),
            "Train F1": round(train_metrics["F1"], 4),
            "Test Accuracy": round(test_metrics["Accuracy"], 4),
            "Test Precision": round(test_metrics["Precision"], 4),
            "Test Recall": round(test_metrics["Recall"], 4),
            "Test F1": round(test_metrics["F1"], 4),
        })
        logger.info("%s -> Test Accuracy: %.4f", name, test_metrics["Accuracy"])

    if not records:
        raise RuntimeError("All model training attempts failed. Check logs above.")

    results_df = pd.DataFrame(records).sort_values(
        by="Test Accuracy", ascending=False
    ).reset_index(drop=True)
    return results_df


def save_best_model(results_df: pd.DataFrame, scaler) -> None:
    """Persist the best model and scaler to disk with joblib."""
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    best_row = results_df.iloc[0]
    best_model = best_row["estimator"]

    joblib.dump(best_model, config.BEST_MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)

    logger.info(
        "Saved best model (%s, Test Accuracy=%.4f) to %s",
        best_row["Model"], best_row["Test Accuracy"], config.BEST_MODEL_PATH,
    )
    logger.info("Saved scaler to %s", config.SCALER_PATH)


def save_metrics_report(results_df: pd.DataFrame) -> None:
    """Save the comparison table (excluding non-serializable estimator objects) to CSV."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    report_df = results_df.drop(columns=["estimator"])
    report_df.to_csv(config.METRICS_PATH, index=False)
    logger.info("Saved metrics report to %s", config.METRICS_PATH)


def run_training_pipeline() -> pd.DataFrame:
    """Full training pipeline: preprocess -> split/scale -> tune -> save."""
    df = run_preprocessing_pipeline(save=True)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)

    results_df = tune_models(X_train, y_train, X_test, y_test)
    print("\n--- Model comparison (sorted by Test Accuracy) ---")
    print(results_df.drop(columns=["estimator"]))

    save_best_model(results_df, scaler)
    save_metrics_report(results_df)

    return results_df


if __name__ == "__main__":
    run_training_pipeline()
