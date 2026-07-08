"""
streamlit_app.py
-----------------
Interactive web app for the Irrigation Water Safety Prediction System.
Users enter water quality parameters and receive a Safe / Unsafe
irrigation prediction along with a model confidence score.

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import sys

import streamlit as st

# Make src/ importable regardless of the working directory this is launched from
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

import config  # noqa: E402
from predict import predict_single  # noqa: E402

st.set_page_config(
    page_title="Irrigation Water Safety Predictor",
    page_icon="💧",
    layout="centered",
)

st.title("💧 Irrigation Water Safety Prediction System")
st.write(
    "Enter water quality parameters below to check whether the water is "
    "**safe for crop irrigation**, along with the model's confidence score."
)

st.divider()

# ---------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("Water Quality Parameters")

    col1, col2 = st.columns(2)

    with col1:
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1,
                              help="Acidity/alkalinity of the water (ideal irrigation range: 6.5 - 8.4)")
        hardness = st.number_input("Hardness (mg/L)", min_value=0.0, value=200.0, step=1.0)
        solids = st.number_input("Solids / TDS (mg/L)", min_value=0.0, value=15000.0, step=100.0,
                                  help="Total Dissolved Solids")
        chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, value=7.0, step=0.1)
        sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, value=330.0, step=1.0)

    with col2:
        conductivity = st.number_input("Conductivity (\u00b5S/cm)", min_value=0.0, value=420.0, step=1.0)
        organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, value=14.0, step=0.1)
        trihalomethanes = st.number_input("Trihalomethanes (\u00b5g/L)", min_value=0.0, value=66.0, step=0.1)
        turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=4.0, step=0.1)

    submitted = st.form_submit_button("Predict Irrigation Safety", use_container_width=True)

# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------
if submitted:
    sample = {
        "ph": ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": sulfate,
        "Conductivity": conductivity,
        "Organic_carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes,
        "Turbidity": turbidity,
    }

    try:
        result = predict_single(sample)
    except FileNotFoundError:
        st.error(
            "Trained model not found. Please run `python src/train_model.py` "
            "from the project root before launching this app."
        )
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")
    else:
        st.divider()
        is_safe = result["prediction_code"] == 1

        if is_safe:
            st.success(f"✅ **{result['prediction']}**")
        else:
            st.error(f"❌ **{result['prediction']}**")

        st.metric("Model Confidence", f"{result['confidence'] * 100:.2f}%")
        st.progress(result["confidence"])

        with st.expander("View submitted parameters"):
            st.json(sample)

st.divider()
st.caption(
    "Model: Random Forest Classifier | Trained on the Water Potability dataset "
    "with domain-calibrated irrigation-safety thresholds for pH, Conductivity, "
    "and Total Dissolved Solids (TDS)."
)
