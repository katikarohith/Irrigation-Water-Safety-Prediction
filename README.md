# 💧 Irrigation Water Safety Prediction System

A machine learning system that predicts whether water is **safe for crop irrigation** based on nine measurable water-quality parameters (pH, hardness, TDS, chloramines, sulfate, conductivity, organic carbon, trihalomethanes, and turbidity). Includes a full ML pipeline (EDA → preprocessing → training → evaluation → prediction) and an interactive Streamlit web app.

---

## 1. Project Overview

Farmers and agricultural planners need a fast way to check whether a water source is suitable for irrigating crops before it damages soil quality or plant health. This project builds a supervised classification model that takes standard water-quality lab readings and outputs:

- **Safe for Irrigation** or **Unsafe for Irrigation**
- A **confidence score** for the prediction

The project was originally developed as an exploratory Jupyter notebook and has been restructured here into a clean, modular, production-style Python codebase suitable for a GitHub portfolio.

---

## 2. Problem Statement

Untreated or poor-quality water used for irrigation can raise soil salinity, damage crops, and reduce long-term farm productivity. Manually checking every water sample against agronomic guidelines (pH range, conductivity, dissolved solids) is slow and error-prone at scale.

**Goal:** Train a classifier that automates this safety check from raw lab measurements, so a user can get an instant Safe/Unsafe decision with a confidence score.

---

## 3. Dataset Description

- **Source:** [Water Potability dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability) (Kaggle) — 3,276 water samples, 9 numeric quality features + a `Potability` column (not used here).
- **Target engineering:** The dataset does not natively include an irrigation-safety label, so one was derived from domain-informed thresholds on three parameters:

  | Condition | Threshold |
  |---|---|
  | pH | between 6.5 and 8.4 |
  | Conductivity | below 600 µS/cm |
  | Total Dissolved Solids (Solids) | below 25,000 mg/L |

  A sample is labeled **Safe (1)** only if *all three* conditions hold, otherwise **Unsafe (0)**.

### Dataset & Labeling Calibration Notes
Standard FAO irrigation-water guidelines commonly cite conductivity < 3000 µS/cm and TDS < 3000 mg/L as safe cut-offs. This dataset's `Conductivity` and `Solids` columns, however, are on a different numeric scale than those raw guideline figures (Conductivity ranges ~180–750 µS/cm; Solids ranges ~320–61,000 mg/L in this dataset). Applying the textbook thresholds directly against this specific dataset would label only 2 of 3,276 rows as "safe" — not usable for training a meaningful classifier. The thresholds above were recalibrated against this dataset's actual distribution, preserving the same guideline logic (lower conductivity/TDS and mid-range pH = safer), producing a workable **73.9% Unsafe / 26.1% Safe** class split. This recalibration is documented here for full transparency; see `src/config.py` for the exact values.

- **Missing values:** `ph` (491 missing → filled with column mean), `Sulfate` (781 missing → column median), `Trihalomethanes` (162 missing → column median).
- **Duplicates:** none found.

---

## 4. Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data handling | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn (Logistic Regression, Random Forest, RandomizedSearchCV) |
| Model persistence | joblib |
| Web app | Streamlit |

---

## 5. ML Workflow

```
data/raw/water_potability.csv
        │
        ▼
src/data_preprocessing.py   → label derivation, missing-value imputation, train/test split, scaling
        │
        ▼
src/eda.py                  → boxplots, distributions, correlation heatmap, class balance (results/figures/)
        │
        ▼
src/train_model.py          → trains Logistic Regression + Random Forest, tunes both via RandomizedSearchCV,
        │                       selects best model by test accuracy, saves model + scaler (models/)
        ▼
src/evaluate_model.py       → classification report, confusion matrix, ROC curve (results/figures/)
        │
        ▼
src/predict.py               → loads saved model to predict on new samples (single or batch)
        │
        ▼
app/streamlit_app.py         → interactive web UI for live predictions
```

---

## 6. Features Used

| Feature | Description | Unit |
|---|---|---|
| `ph` | Acidity/alkalinity | 0–14 scale |
| `Hardness` | Calcium/magnesium salt concentration | mg/L |
| `Solids` | Total Dissolved Solids (TDS) | mg/L |
| `Chloramines` | Disinfectant concentration | ppm |
| `Sulfate` | Sulfate concentration | mg/L |
| `Conductivity` | Electrical conductivity | µS/cm |
| `Organic_carbon` | Organic carbon content | ppm |
| `Trihalomethanes` | Disinfection by-product concentration | µg/L |
| `Turbidity` | Water cloudiness | NTU |

Target: `Irrigation_Safety` (1 = Safe, 0 = Unsafe) — derived, see Section 3.

---

## 7. Models Trained

Two classifiers were trained and tuned with `RandomizedSearchCV` (3-fold CV, 20 parameter combinations):

1. **Logistic Regression** — linear baseline
2. **Random Forest Classifier** — ensemble tree-based model

The Random Forest substantially outperformed Logistic Regression because the true labeling rule involves non-linear threshold interactions across features — exactly the kind of pattern tree ensembles capture well, while a linear decision boundary struggles with.

---

## 8. Evaluation Metrics

Results on the held-out 30% test set (983 samples), from `results/model_metrics.csv`:

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 |
|---|---|---|---|---|
| **Random Forest (best)** | **0.9929** | **1.0000** | **0.9728** | **0.9862** |
| Logistic Regression | 0.7345 | 0.4091 | 0.0350 | 0.0645 |

**Random Forest — Classification Report:**

```
                       precision    recall  f1-score   support

Unsafe for Irrigation       0.99      1.00      1.00       726
  Safe for Irrigation       1.00      0.97      0.99       257

             accuracy                           0.99       983
            macro avg       1.00      0.99      0.99       983
         weighted avg       0.99      0.99      0.99       983
```

ROC-AUC: **1.000**

Figures (confusion matrix, ROC curve, correlation heatmap, class balance, boxplots) are saved in `results/figures/` and regenerated automatically by `src/eda.py` / `src/evaluate_model.py`.

---

## 9. Project Structure

```
irrigation-water-safety-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── app/
│   └── streamlit_app.py          # Streamlit web application
├── data/
│   ├── raw/
│   │   └── water_potability.csv  # original dataset
│   └── processed/
│       └── processed_data.csv    # cleaned + labeled dataset (generated)
├── models/
│   ├── best_model.joblib         # trained Random Forest model
│   └── scaler.joblib             # fitted StandardScaler
├── notebooks/
│   └── irrigation_safety_original_analysis.ipynb   # original exploratory notebook
├── results/
│   ├── model_metrics.csv
│   └── figures/                  # EDA + evaluation plots
└── src/
    ├── config.py                 # paths, feature lists, thresholds
    ├── data_preprocessing.py     # loading, labeling, cleaning, split/scale
    ├── eda.py                    # exploratory data analysis + plots
    ├── train_model.py            # model training + hyperparameter tuning
    ├── evaluate_model.py         # evaluation report + plots
    └── predict.py                # single/batch prediction functions
```

---

## 10. How to Run the Project

### 10.1 Setup

```bash
git clone <your-repo-url>
cd irrigation-water-safety-prediction
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 10.2 Run the pipeline (in order)

```bash
# 1. Clean & preprocess the raw data
python src/data_preprocessing.py

# 2. Generate EDA plots
python src/eda.py

# 3. Train and tune models, save the best one
python src/train_model.py

# 4. Evaluate the saved model
python src/evaluate_model.py

# 5. Run a quick CLI prediction demo
python src/predict.py
```

### 10.3 Launch the web app

```bash
streamlit run app/streamlit_app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`) in your browser.

---

## 11. Sample Prediction Examples

Using `src/predict.py`'s `predict_single()` function:

**Sample A — expected Safe**
```python
sample = {
    "ph": 7.2, "Hardness": 200.0, "Solids": 15000.0, "Chloramines": 6.5,
    "Sulfate": 300.0, "Conductivity": 450.0, "Organic_carbon": 12.0,
    "Trihalomethanes": 60.0, "Turbidity": 3.5,
}
```
**Output:** `Safe for Irrigation` — Confidence: **93.03%**

**Sample B — expected Unsafe**
```python
sample = {
    "ph": 5.1, "Hardness": 180.0, "Solids": 32000.0, "Chloramines": 7.8,
    "Sulfate": 280.0, "Conductivity": 3400.0, "Organic_carbon": 15.0,
    "Trihalomethanes": 70.0, "Turbidity": 4.1,
}
```
**Output:** `Unsafe for Irrigation` — Confidence: **99.80%**

Both examples were generated by actually running `python src/predict.py`.

---

## 12. Known Limitations & Future Improvements

- The `Irrigation_Safety` label is a **rule-derived** target (not ground-truth field data), so the model is ultimately learning the labeling rule rather than a real-world outcome. A dataset with actual crop-yield or soil-health outcomes would make this more robust.
- Threshold values were recalibrated to this dataset's scale (see Section 3) — for a production deployment, thresholds should be validated against local agronomic standards for the target region and crop type.
- Class imbalance (73.9% / 26.1%) means precision/recall on the minority ("Safe") class deserves more attention than raw accuracy alone; this is why the evaluation report above includes per-class precision/recall rather than accuracy only.
- Next steps: incorporate additional parameters (sodium adsorption ratio, boron, chloride), try gradient boosting models (XGBoost/LightGBM), and validate against real field outcomes.

---

## 13. Author / Resume Notes

This project demonstrates an end-to-end applied ML workflow: data cleaning, feature engineering, exploratory data analysis, model comparison, hyperparameter tuning, evaluation, and deployment via an interactive web app — relevant to Data Analyst, Business Analyst, and ML fresher roles.
