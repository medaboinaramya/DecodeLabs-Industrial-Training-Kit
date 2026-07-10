"""
app.py
------
Project 2: Data Classification Using AI (DecodeLabs)
Streamlit front-end for the Iris KNN classifier.

Run with:
    streamlit run app.py

Note: this app expects 'iris_model.pkl' to exist in the same folder.
If it's missing, the app will train one automatically on first run.
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from model import (
    load_data,
    build_pipeline,
    evaluate,
    save_artifacts,
)

MODEL_PATH = "iris_model.pkl"

st.set_page_config(
    page_title="Iris Data Classification | DecodeLabs",
    page_icon="🌸",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Model loading (train once, cache after that)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_artifacts():
    if not os.path.exists(MODEL_PATH):
        X, y, feature_names, target_names = load_data()
        model, scaler, X_train, X_test, y_train, y_test = build_pipeline(X, y, n_neighbors=5)
        save_artifacts(model, scaler, feature_names, target_names, path=MODEL_PATH)

    with open(MODEL_PATH, "rb") as f:
        artifacts = pickle.load(f)
    return artifacts


@st.cache_data
def get_eval_data(_model, _scaler, n_neighbors):
    """Re-run the split/eval so we can show a confusion matrix & F1 score."""
    X, y, feature_names, target_names = load_data()
    model, scaler, X_train, X_test, y_train, y_test = build_pipeline(X, y, n_neighbors=n_neighbors)
    results = evaluate(model, X_test, y_test, target_names)
    return results, target_names


artifacts = get_artifacts()
model = artifacts["model"]
scaler = artifacts["scaler"]
feature_names = artifacts["feature_names"]
target_names = artifacts["target_names"]

FLOWER_EMOJI = {"setosa": "🌱", "versicolor": "🌷", "virginica": "🌺"}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🌸 Iris Data Classification Using AI")
st.caption("Project 2 — Industrial Training Kit — DecodeLabs")

st.markdown(
    """
This app trains a **K-Nearest Neighbors (KNN)** classifier on the classic
**Iris dataset** (150 samples, 3 classes, 4 features) and lets you classify
a new flower by entering its measurements — the same supervised learning
pipeline from the training kit: **Input → Process → Output**.
"""
)

tab_predict, tab_model = st.tabs(["🔮 Predict", "📊 Model Performance"])

# ---------------------------------------------------------------------------
# TAB 1: Prediction (this is the "index.html" -> "result.html" flow)
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("Enter Flower Measurements")

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 3.7, 0.1)
    with col2:
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2, 0.1)

    if st.button("Classify Flower", type="primary", use_container_width=True):
        input_df = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=feature_names,
        )
        scaled_input = scaler.transform(input_df)

        prediction = model.predict(scaled_input)[0]
        probabilities = model.predict_proba(scaled_input)[0]
        predicted_class = target_names[prediction]

        st.divider()
        st.markdown("### Result")
        emoji = FLOWER_EMOJI.get(predicted_class, "🌼")
        st.success(f"{emoji} Predicted species: **{predicted_class.capitalize()}**")

        prob_df = pd.DataFrame(
            {"Species": [t.capitalize() for t in target_names], "Confidence": probabilities}
        ).sort_values("Confidence", ascending=False)

        st.bar_chart(prob_df.set_index("Species"))
        st.dataframe(
            prob_df.style.format({"Confidence": "{:.1%}"}),
            use_container_width=True,
            hide_index=True,
        )

# ---------------------------------------------------------------------------
# TAB 2: Model performance (confusion matrix, F1, accuracy)
# ---------------------------------------------------------------------------
with tab_model:
    st.subheader("Output Validation")
    st.caption("In imbalanced data, accuracy alone is a mirage — we look deeper.")

    n_neighbors = st.slider("K (number of neighbors)", 1, 20, 5)
    results, tn = get_eval_data(model, scaler, n_neighbors)

    col_a, col_b = st.columns(2)
    col_a.metric("Accuracy", f"{results['accuracy']:.2%}")
    col_b.metric("F1 Score (weighted)", f"{results['f1_score']:.2%}")

    st.markdown("#### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=results["confusion_matrix"],
        display_labels=[t.capitalize() for t in tn],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    st.pyplot(fig)

    st.markdown("#### Classification Report")
    report_df = pd.DataFrame(results["classification_report"]).transpose()
    st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)

st.divider()
st.caption("DecodeLabs — Artificial Intelligence Industrial Training Kit — Batch 2026")
