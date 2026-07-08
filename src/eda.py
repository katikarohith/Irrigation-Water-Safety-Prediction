"""
eda.py
------
Exploratory Data Analysis for the Irrigation Water Safety dataset.
Generates and saves the same visuals produced in the original notebook
(boxplots, distribution histograms, correlation heatmap, pairplot,
class balance) into results/figures/, plus a printed summary of key
statistics to the console.

Run directly:
    python src/eda.py
"""

import os
import logging

import matplotlib
matplotlib.use("Agg")  # headless backend so this runs without a display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import config
from data_preprocessing import run_preprocessing_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _save_fig(fig, filename: str) -> None:
    """Save a matplotlib figure into results/figures/ and close it."""
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    path = os.path.join(config.FIGURES_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.info("Saved figure: %s", path)


def summarize_dataset(df: pd.DataFrame) -> None:
    """Print shape, dtypes, missing values, and class balance to console."""
    print("\n--- Dataset shape ---")
    print(df.shape)

    print("\n--- Missing values ---")
    print(df.isnull().sum())

    print("\n--- Duplicate rows ---")
    print(df.duplicated().sum())

    print("\n--- Summary statistics ---")
    print(df.describe())

    print("\n--- Target class balance ---")
    print(df[config.TARGET_COLUMN].value_counts(normalize=True).round(3))


def plot_boxplots(df: pd.DataFrame) -> None:
    """One boxplot per feature to visualize outliers."""
    features = [c for c in df.columns if c not in (config.TARGET_COLUMN, "Potability")]
    n_cols = 3
    n_rows = -(-len(features) // n_cols)  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for ax, feature in zip(axes, features):
        sns.boxplot(x=df[feature], ax=ax)
        ax.set_title(feature)

    for ax in axes[len(features):]:
        ax.axis("off")

    fig.tight_layout()
    _save_fig(fig, "boxplots_all_features.png")


def plot_missing_value_distributions(df: pd.DataFrame) -> None:
    """Histograms for the columns that originally contained missing values."""
    features = list(config.IMPUTATION_STRATEGY.keys())
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4))
    if len(features) == 1:
        axes = [axes]

    for ax, feature in zip(axes, features):
        sns.histplot(x=df[feature], kde=True, ax=ax)
        ax.set_title(feature)

    fig.tight_layout()
    _save_fig(fig, "distribution_missing_value_features.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Correlation heatmap across all numeric features."""
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    _save_fig(fig, "correlation_heatmap.png")


def plot_target_balance(df: pd.DataFrame) -> None:
    """Bar chart of Safe vs Unsafe irrigation class counts."""
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[config.TARGET_COLUMN].value_counts().sort_index()
    labels = [config.LABEL_MAP[i] for i in counts.index]
    sns.barplot(x=labels, y=counts.values, hue=labels, ax=ax, palette="viridis", legend=False)
    ax.set_ylabel("Count")
    ax.set_title("Irrigation Safety Class Balance")
    _save_fig(fig, "target_class_balance.png")


def run_eda() -> None:
    """Execute the full EDA workflow end-to-end."""
    df = run_preprocessing_pipeline(save=False)
    summarize_dataset(df)
    plot_boxplots(df)
    plot_missing_value_distributions(df)
    plot_correlation_heatmap(df)
    plot_target_balance(df)
    logger.info("EDA complete. Figures saved to %s", config.FIGURES_DIR)


if __name__ == "__main__":
    run_eda()
