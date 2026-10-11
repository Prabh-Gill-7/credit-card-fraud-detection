"""
Test Prediction Script for Credit Card Fraud Detection
======================================================
Loads serialized models and test samples to perform instant real-time predictions.

Usage:
    python scripts/test_prediction.py
"""

import os
import joblib
import pandas as pd

def resolve_path(relative_path):
    """Resolves path relative to project root or current working directory."""
    if os.path.exists(relative_path):
        return relative_path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fallback_path = os.path.join(base_dir, relative_path)
    if os.path.exists(fallback_path):
        return fallback_path
    return relative_path

def predict_transaction(
    transaction,
    model_path=None,
    scaler_path=None,
    feature_cols_path=None,
    cols_to_scale=("Time", "Amount")
):
    """Predict whether a transaction vector is legitimate or fraudulent."""
    model_path = model_path or resolve_path(os.path.join("models", "fraud_detection_model.pkl"))
    scaler_path = scaler_path or resolve_path(os.path.join("models", "scaler.pkl"))
    feature_cols_path = feature_cols_path or resolve_path(os.path.join("models", "feature_columns.pkl"))

    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(feature_cols_path)):
        raise FileNotFoundError(
            f"Model artifacts not found at:\n- {model_path}\n- {scaler_path}\n- {feature_cols_path}\n"
            "Run the notebook or 'scripts/fraud_detection.py' first."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(feature_cols_path)

    # Convert to DataFrame matching training columns
    input_df = pd.DataFrame([transaction], columns=feature_columns)

    # Scale Time and Amount
    cols_list = list(cols_to_scale)
    input_df[cols_list] = scaler.transform(input_df[cols_list])

    # Predict
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    result_text = "FRAUDULENT TRANSACTION" if prediction == 1 else "LEGITIMATE TRANSACTION"

    print("\n" + "=" * 45)
    print("           TRANSACTION RESULT")
    print("=" * 45)
    print(f"Prediction        : {result_text}")
    print(f"Fraud Probability : {round(probability * 100, 2)} %")
    print("=" * 45)

    return prediction, probability


def main():
    csv_file = resolve_path(os.path.join("data", "creditcard.csv"))

    if not os.path.exists(csv_file):
        print(f"Error: '{csv_file}' not found.")
        print("Please ensure 'creditcard.csv' is placed in 'data/' directory.")
        return

    print(f"Reading test samples from: '{csv_file}'...")
    df = pd.read_csv(csv_file)

    # Test a Legitimate sample (Class 0)
    legit_subset = df[df["Class"] == 0]
    if not legit_subset.empty:
        print("\n>>> Testing Sample 1: Known LEGITIMATE Transaction (Class 0)")
        sample_legit = legit_subset.drop("Class", axis=1).iloc[0].tolist()
        predict_transaction(sample_legit)

    # Test a Fraudulent sample (Class 1)
    fraud_subset = df[df["Class"] == 1]
    if not fraud_subset.empty:
        print("\n>>> Testing Sample 2: Known FRAUDULENT Transaction (Class 1)")
        sample_fraud = fraud_subset.drop("Class", axis=1).iloc[0].tolist()
        predict_transaction(sample_fraud)
    else:
        print("\nNote: No fraud transaction found in dataset to test.")


if __name__ == "__main__":
    main()
