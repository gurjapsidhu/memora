


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ---------------- CONFIG ----------------

st.set_page_config(
    page_title="MEMORA | Research Studio",
    page_icon="🧠",
    layout="wide"
)

BASE = Path(__file__).resolve().parent

# ---------------- LOAD FILES ----------------

@st.cache_resource
def load_model():
    return joblib.load(BASE / "gb_model.pkl")

@st.cache_data
def load_data():
    X = pd.read_csv(BASE / "X_test.csv")
    y = pd.read_csv(BASE / "Y_test.csv").iloc[:, 0]
    return X, y

try:
    model = load_model()
    X_test, Y_test = load_data()

except Exception as e:
    st.error("Unable to load model or test data.")
    st.exception(e)
    st.stop()

if X_test.shape[1] != 49:
    st.error("The dataset must contain exactly 49 features.")
    st.stop()

features = list(X_test.columns)

# ---------------- PREDICTIONS ----------------

predictions = model.predict(X_test)

r2 = model.score(X_test, Y_test)
mae = np.mean(np.abs(Y_test - predictions))
rmse = np.sqrt(np.mean((Y_test - predictions) ** 2))

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.title("🧠 MEMORA")
    st.caption("OBJECT MEMORY RESEARCH")

    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Custom Prediction",
            "Test Samples",
            "Model Analytics",
            "Research Notes"
        ]
    )

    st.divider()

    st.write("**MODEL**")
    st.write("Gradient Boosting Regressor")
    st.caption("49 semantic dimensions")

    st.divider()

    st.write("**DATASET**")
    st.write("THINGS Memorability")

# ---------------- HEADER ----------------

st.title("MEMORA")
st.write("Semantic representations and human object memorability.")

st.divider()

# ==================================================
# OVERVIEW
# ==================================================

if page == "Overview":

    st.header("Research Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Test Observations", f"{len(X_test):,}")
    c2.metric("Input Dimensions", "49")
    c3.metric("R² Score", f"{r2:.4f}")
    c4.metric("Mean Absolute Error", f"{mae:.4f}")

    st.divider()

    st.subheader("Human vs Model Predictions")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=Y_test,
        y=predictions,
        mode="markers",
        name="Predictions"
    ))

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Ideal Prediction"
    ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Human Memorability Score",
        yaxis_title="Predicted Score",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("About the Project")

    st.write("""
    MEMORA explores the relationship between semantic
    representations and human object memorability.

    The Gradient Boosting model uses 49 semantic dimensions
    to predict Corrected Recognition (cr).
    """)

# ==================================================
# CUSTOM PREDICTION
# ==================================================

elif page == "Custom Prediction":

    st.header("Custom Prediction")

    st.write(
        "Enter values for the 49 semantic dimensions "
        "to generate a memorability estimate."
    )

    st.info(
        "Use values from the same feature extraction "
        "and preprocessing pipeline used to train the model."
    )

    with st.form("prediction_form"):

        values = []

        for i in range(0, 49, 7):

            st.subheader(
                f"Dimensions {i + 1} to {min(i + 7, 49)}"
            )

            cols = st.columns(4)

            for j, feature in enumerate(features[i:i + 7]):

                with cols[j % 4]:

                    value = st.number_input(
                        feature,
                        value=0.0,
                        format="%.6f"
                    )

                    values.append(value)

        submitted = st.form_submit_button(
            "Generate Prediction",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame(
            [values],
            columns=features
        )

        result = float(model.predict(input_data)[0])

        st.divider()

        st.subheader("Prediction Result")

        st.metric(
            "Predicted Memorability Score",
            f"{result:.4f}"
        )

        st.caption(
            "This is a model estimate, not a human recognition score."
        )

        st.download_button(
            "Download Input Values",
            input_data.to_csv(index=False),
            file_name="MEMORA_input.csv",
            mime="text/csv"
        )

# ==================================================
# TEST SAMPLES
# ==================================================

elif page == "Test Samples":

    st.header("Test Sample Explorer")

    index = st.slider(
        "Select a test observation",
        0,
        len(X_test) - 1,
        0
    )

    actual = float(Y_test.iloc[index])
    predicted = float(predictions[index])

    c1, c2, c3 = st.columns(3)

    c1.metric("Human Score", f"{actual:.4f}")
    c2.metric("Predicted Score", f"{predicted:.4f}")
    c3.metric("Absolute Error", f"{abs(actual-predicted):.4f}")

    st.divider()

    st.subheader("Human vs Model")

    comparison = pd.DataFrame({
        "Type": ["Human Score", "Model Prediction"],
        "Score": [actual, predicted]
    })

    fig = px.bar(
        comparison,
        x="Type",
        y="Score",
        title="Score Comparison"
    )

    fig.update_layout(
        template="plotly_dark",
        yaxis_range=[0, 1]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("49 Semantic Dimensions")

    st.dataframe(
        X_test.iloc[[index]].T.rename(
            columns={index: "Value"}
        ),
        use_container_width=True
    )

# ==================================================
# MODEL ANALYTICS
# ==================================================

elif page == "Model Analytics":

    st.header("Model Analytics")

    c1, c2, c3 = st.columns(3)

    c1.metric("R² Score", f"{r2:.4f}")
    c2.metric("MAE", f"{mae:.4f}")
    c3.metric("RMSE", f"{rmse:.4f}")

    st.divider()

    st.subheader("Residual Distribution")

    residuals = Y_test.to_numpy() - predictions

    fig = px.histogram(
        x=residuals,
        nbins=40,
        labels={"x": "Residual"}
    )

    fig.update_layout(
        template="plotly_dark",
        yaxis_title="Frequency"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Feature Importance")

    if hasattr(model, "feature_importances_"):

        importance = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        })

        importance = importance.nlargest(
            15, "Importance"
        ).sort_values("Importance")

        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h"
        )

        fig.update_layout(
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Feature importance does not establish causation."
        )

    else:
        st.info("Feature importance is unavailable.")

    results = pd.DataFrame({
        "Actual": Y_test,
        "Predicted": predictions,
        "Residual": residuals
    })

    st.download_button(
        "Download Prediction Results",
        results.to_csv(index=False),
        file_name="MEMORA_results.csv",
        mime="text/csv"
    )

# ==================================================
# RESEARCH NOTES
# ==================================================

elif page == "Research Notes":

    st.header("Research Framework")

    st.subheader("Central Research Question")

    st.write(
        "Does an object's semantic distinctiveness "
        "help explain its memorability?"
    )

    st.divider()

    st.subheader("Dataset")

    st.write("""
    - THINGS Memorability
    - 26,106 usable observations
    - 49 semantic dimensions
    - Target variable: Corrected Recognition (cr)
    """)

    st.divider()

    st.subheader("Model")

    st.write("""
    - Gradient Boosting Regressor
    - Input: dim_1 to dim_49
    - Output: Predicted Corrected Recognition score
    """)

    st.divider()

    st.subheader("Current Performance")

    st.write(f"R² Score: {r2:.4f}")
    st.write(f"Mean Absolute Error: {mae:.4f}")
    st.write(f"RMSE: {rmse:.4f}")

    st.divider()

    st.subheader("Limitations")

    st.write("""
    1. The model requires 49 semantic dimensions.
    2. It does not directly accept image uploads.
    3. Semantic dimensions do not capture every visual property.
    4. Predictions are estimates, not actual human memory tests.
    """)

# ---------------- FOOTER ----------------

st.divider()

st.caption("MEMORA | Intelligence Research Studio | 2026")
