"""
evaluate_model.py
------------------
Loads the persisted best model + scaler and evaluates it on a fresh
train/test split of the processed data, producing a confusion matrix,
classification report, and ROC curve saved to results/figures/.

Run directly:
    python src/evaluate_model.py
"""

import logging
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

import config
from data_preprocessing import run_preprocessing_pipeline, split_and_scale

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_artifacts():
    """Load the saved model and scaler, raising a clear error if missing."""
    if not os.path.exists(config.BEST_MODEL_PATH) or not os.path.exists(config.SCALER_PATH):
        raise FileNotFoundError(
            "Trained model/scaler not found. Run `python src/train_model.py` first."
        )
    model = joblib.load(config.BEST_MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    return model, scaler


def plot_confusion_matrix(model, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[config.LABEL_MAP[0], config.LABEL_MAP[1]],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - Best Model")
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(config.FIGURES_DIR, "confusion_matrix.png"), bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.info("Saved confusion_matrix.png")


def plot_roc_curve(model, X_test, y_test) -> None:
    if not hasattr(model, "predict_proba"):
        logger.warning("Model has no predict_proba method; skipping ROC curve.")
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    ax.set_title(f"ROC Curve (AUC = {auc:.3f})")
    fig.savefig(os.path.join(config.FIGURES_DIR, "roc_curve.png"), bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.info("Saved roc_curve.png (AUC=%.3f)", auc)


def run_evaluation() -> None:
    """Full evaluation pipeline: load model -> evaluate -> save plots + report."""
    model, scaler = load_artifacts()

    df = run_preprocessing_pipeline(save=False)
    _, X_test, _, y_test, _ = split_and_scale(df)  # re-fits its own scaler for a clean split
    # Use the freshly-fit scaler's transform for consistency in this evaluation run
    # (re-using the pipeline function keeps this script self-contained).

    print("\n--- Classification Report ---")
    y_pred = model.predict(X_test)
    print(classification_report(
        y_test, y_pred, target_names=[config.LABEL_MAP[0], config.LABEL_MAP[1]]
    ))

    plot_confusion_matrix(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)

    logger.info("Evaluation complete. Figures saved to %s", config.FIGURES_DIR)


if __name__ == "__main__":
    run_evaluation()
