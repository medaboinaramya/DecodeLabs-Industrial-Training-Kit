"""
model.py
--------
Project 2: Data Classification Using AI (DecodeLabs)

Builds the full IPO pipeline described in the training kit:
  INPUT   -> Load Iris dataset, scale features
  PROCESS -> Train/test split, K-Nearest Neighbors algorithm
  OUTPUT  -> Confusion matrix, F1 score, saved model artifacts

Run this once to train and save the model:
    python model.py
"""

import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
)


def load_data():
    """INPUT stage: load the Iris benchmark dataset (150 samples, 3 classes, 4 features)."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    return X, y, feature_names, target_names


def build_pipeline(X, y, n_neighbors=5, test_size=0.2, random_state=42):
    """PROCESS stage: scale, split, and train the KNN classifier."""

    # The Gatekeeper Rule: Scaling -> mean 0, variance 1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Structural Integrity: The Split (shuffle removes order bias)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y
    )

    # The Algorithm: K-Nearest Neighbors
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    return model, scaler, X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test, target_names):
    """OUTPUT stage: confusion matrix, F1 score, accuracy (avoiding the 'accuracy mirage')."""
    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=target_names, output_dict=True)
    f1 = f1_score(y_test, predictions, average="weighted")
    acc = accuracy_score(y_test, predictions)

    return {
        "confusion_matrix": cm,
        "classification_report": report,
        "f1_score": f1,
        "accuracy": acc,
        "predictions": predictions,
    }


def find_best_k(X, y, k_range=range(1, 21), test_size=0.2, random_state=42):
    """Tuning the Engine: sweep K values to find the 'elbow' (optimal K)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y
    )

    errors = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        error_rate = np.mean(pred != y_test)
        errors.append(error_rate)

    best_k = list(k_range)[int(np.argmin(errors))]
    return best_k, errors


def save_artifacts(model, scaler, feature_names, target_names, path="iris_model.pkl"):
    """Persist model + scaler + metadata for the Streamlit app to load."""
    artifacts = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "target_names": list(target_names),
    }
    with open(path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"Saved trained model + scaler to '{path}'")


if __name__ == "__main__":
    X, y, feature_names, target_names = load_data()

    # Find the elbow (optimal K) before final training
    best_k, errors = find_best_k(X, y)
    print(f"Optimal K found via elbow search: {best_k}")

    model, scaler, X_train, X_test, y_train, y_test = build_pipeline(X, y, n_neighbors=best_k)
    results = evaluate(model, X_test, y_test, target_names)

    print("\n--- Confusion Matrix ---")
    print(results["confusion_matrix"])
    print(f"\nAccuracy: {results['accuracy']:.4f}")
    print(f"F1 Score (weighted): {results['f1_score']:.4f}")

    save_artifacts(model, scaler, feature_names, target_names)
