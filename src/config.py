"""
config.py
---------
Central configuration for the Irrigation Water Safety Prediction project.
Keeping paths, feature lists, and thresholds here avoids "magic values"
scattered across scripts and makes the project easy to maintain.
"""

import os

# ---------------------------------------------------------------------
# Directory paths (all relative to project root)
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

RAW_DATA_PATH = os.path.join(DATA_RAW_DIR, "water_potability.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_PROCESSED_DIR, "processed_data.csv")

# ---------------------------------------------------------------------
# Model & scaler artifact paths
# ---------------------------------------------------------------------
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
METRICS_PATH = os.path.join(RESULTS_DIR, "model_metrics.csv")

# ---------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------
TARGET_COLUMN = "Irrigation_Safety"

# Raw feature columns used to train the model (Potability is dropped —
# it is a *different* label, unrelated to irrigation safety).
FEATURE_COLUMNS = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

# Columns that contain missing values in the raw dataset, and the
# strategy used to impute each one (mirrors the original notebook).
IMPUTATION_STRATEGY = {
    "ph": "mean",
    "Sulfate": "median",
    "Trihalomethanes": "median",
}

# ---------------------------------------------------------------------
# Domain thresholds used to derive the Irrigation_Safety target label.
# Water is considered SAFE for irrigation (1) when ALL conditions hold;
# otherwise it is labeled UNSAFE (0).
#
# NOTE ON CALIBRATION: FAO irrigation-water guidelines commonly cite
# Conductivity < 3000 uS/cm and TDS < 3000 mg/L as safe cut-offs.
# However, this dataset's 'Solids' feature ranges ~320-61,000 (mean
# ~22,000), and 'Conductivity' ranges ~180-750 -- both on a different
# scale than the raw FAO figures. Applying the raw thresholds directly
# leaves only 2 "safe" rows out of 3,276, which is unusable for
# training. The thresholds below are calibrated to this dataset's
# actual distribution (using the same guideline ordering/intent -
# lower conductivity & TDS, mid-range pH = safer) so the label carries
# real signal. See README "Dataset & Labeling Notes" for details.
# ---------------------------------------------------------------------
CONDUCTIVITY_MAX = 600        # micro-Siemens/cm (dataset-calibrated)
TDS_MAX = 25000                # Total Dissolved Solids, mg/L (dataset-calibrated)
PH_MIN = 6.5
PH_MAX = 8.4

# ---------------------------------------------------------------------
# Train / test split & reproducibility
# ---------------------------------------------------------------------
TEST_SIZE = 0.3
RANDOM_STATE = 42

LABEL_MAP = {0: "Unsafe for Irrigation", 1: "Safe for Irrigation"}
