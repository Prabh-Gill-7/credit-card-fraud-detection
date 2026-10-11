# ============================================================
# CREDIT CARD FRAUD DETECTION USING MACHINE LEARNING
# ============================================================
# Complete Training, Evaluation, and Serialization Pipeline

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

from imblearn.over_sampling import SMOTE
import joblib


def resolve_path(relative_path):
    """Resolves path relative to current working directory or project root."""
    if os.path.exists(relative_path):
        return relative_path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fallback_path = os.path.join(base_dir, relative_path)
    if os.path.exists(fallback_path):
        return fallback_path
    return relative_path


# ============================================================
# Prediction Function (Callable independently)
# ============================================================

def predict_transaction(transaction, model_path=None, scaler_path=None,
                        feature_cols_path=None, cols_to_scale=("Time", "Amount")):
    """Scores a single transaction against the serialized model."""
    model_path = model_path or resolve_path(os.path.join("models", "fraud_detection_model.pkl"))
    scaler_path = scaler_path or resolve_path(os.path.join("models", "scaler.pkl"))
    feature_cols_path = feature_cols_path or resolve_path(os.path.join("models", "feature_columns.pkl"))

    model = joblib.load(model_path)
    saved_scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(feature_cols_path)

    # Convert input to DataFrame
    input_data = pd.DataFrame([transaction], columns=feature_columns)

    # Scale Time and Amount
    cols_list = list(cols_to_scale)
    input_data[cols_list] = saved_scaler.transform(input_data[cols_list])

    # Prediction
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    result = "FRAUDULENT TRANSACTION" if prediction == 1 else "LEGITIMATE TRANSACTION"

    print("\n================ TRANSACTION RESULT ================\n")
    print("Prediction:", result)
    print("Fraud Probability:", round(probability * 100, 2), "%")

    return prediction, probability


# ============================================================
# Main Training & Evaluation Pipeline
# ============================================================

def run_pipeline():
    # Setup directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "scripts" in __file__ else "."
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    vis_dir = os.path.join(base_dir, "visualizations")

    for d in [data_dir, models_dir, vis_dir]:
        os.makedirs(d, exist_ok=True)

    csv_candidates = [
        os.path.join(data_dir, "creditcard.csv"),
        "creditcard.csv",
        os.path.join("data", "creditcard.csv")
    ]
    file_path = next((p for p in csv_candidates if os.path.exists(p)), None)

    if not file_path:
        raise FileNotFoundError(
            "creditcard.csv not found in data/ or project root.\n"
            "Download from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )

    print(f"Loading dataset from '{file_path}'...")
    df = pd.read_csv(file_path)

    print("\n================ DATASET INFORMATION ================\n")
    print("Shape of dataset:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumn names:", df.columns.tolist())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nClass distribution:\n", df["Class"].value_counts())

    # -----------------------------
    # EDA
    # -----------------------------
    print("\n================ EDA ================\n")
    class_counts = df["Class"].value_counts()
    legitimate = class_counts.get(0, 0)
    fraud = class_counts.get(1, 0)
    print(f"Legitimate transactions: {legitimate}")
    print(f"Fraudulent transactions: {fraud}")
    print(f"Fraud percentage: {round((fraud / len(df)) * 100, 4)} %")

    # Class Distribution Plot
    plt.figure(figsize=(7, 5))
    sns.countplot(x="Class", data=df, hue="Class", palette=["green", "red"], legend=False)
    plt.title("Transaction Class Distribution")
    plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
    plt.ylabel("Number of Transactions")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "class_distribution.png"))
    plt.close()
    print("Saved plot: visualizations/class_distribution.png")

    # Amount Distribution Plot
    plt.figure(figsize=(10, 5))
    sns.histplot(df["Amount"], bins=50, kde=True)
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Transaction Amount")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "amount_distribution.png"))
    plt.close()
    print("Saved plot: visualizations/amount_distribution.png")

    # Boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Class", y="Amount", data=df, hue="Class", palette=["green", "red"], legend=False)
    plt.title("Transaction Amount: Legitimate vs Fraud")
    plt.xlabel("Class")
    plt.ylabel("Amount")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "fraud_amount_boxplot.png"))
    plt.close()
    print("Saved plot: visualizations/fraud_amount_boxplot.png")

    # -----------------------------
    # Preprocessing
    # -----------------------------
    print("\n================ PREPROCESSING ================\n")
    duplicate_count = df.duplicated().sum()
    print("Duplicate rows:", duplicate_count)
    df = df.drop_duplicates()
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
    print("Shape after cleaning:", df.shape)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    # Scaling
    scaler = StandardScaler()
    columns_to_scale = ["Time", "Amount"]
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

    # SMOTE
    print("\n================ BEFORE SMOTE ================\n")
    print(y_train.value_counts())
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print("\n================ AFTER SMOTE ================\n")
    print(y_train_smote.value_counts())

    # -----------------------------
    # Models Training
    # -----------------------------
    print("\n================ LOGISTIC REGRESSION ================\n")
    logistic_model = LogisticRegression(max_iter=1000, random_state=42)
    logistic_model.fit(X_train_smote, y_train_smote)
    logistic_prediction = logistic_model.predict(X_test)
    logistic_probability = logistic_model.predict_proba(X_test)[:, 1]

    print("\n================ RANDOM FOREST ================\n")
    random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    random_forest_model.fit(X_train_smote, y_train_smote)
    rf_prediction = random_forest_model.predict(X_test)
    rf_probability = random_forest_model.predict_proba(X_test)[:, 1]

    # Evaluation Helper
    def evaluate_model(name, y_true, y_pred, y_prob):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob)

        print("\n----------------------------------------")
        print(name)
        print("----------------------------------------")
        print("Accuracy :", round(acc, 4))
        print("Precision:", round(prec, 4))
        print("Recall   :", round(rec, 4))
        print("F1 Score :", round(f1, 4))
        print("ROC-AUC  :", round(auc, 4))
        print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=0))
        print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))

        return {
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC-AUC": auc
        }

    log_res = evaluate_model("Logistic Regression", y_test, logistic_prediction, logistic_probability)
    rf_res = evaluate_model("Random Forest", y_test, rf_prediction, rf_probability)

    results_df = pd.DataFrame([log_res, rf_res])
    print("\n================ MODEL COMPARISON ================\n")
    print(results_df)

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, rf_prediction)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legitimate", "Fraud"], yticklabels=["Legitimate", "Fraud"])
    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "confusion_matrix.png"))
    plt.close()
    print("Saved plot: visualizations/confusion_matrix.png")

    # ROC Curve Plot
    l_fpr, l_tpr, _ = roc_curve(y_test, logistic_probability)
    r_fpr, r_tpr, _ = roc_curve(y_test, rf_probability)
    plt.figure(figsize=(8, 6))
    plt.plot(l_fpr, l_tpr, label="Logistic Regression")
    plt.plot(r_fpr, r_tpr, label="Random Forest")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "roc_curve.png"))
    plt.close()
    print("Saved plot: visualizations/roc_curve.png")

    # Feature Importance
    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": random_forest_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    print("\n================ FEATURE IMPORTANCE ================\n")
    print(feature_importance.head(15))

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(15), x="Importance", y="Feature", palette="viridis")
    plt.title("Top 15 Important Features")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "feature_importance.png"))
    plt.close()
    print("Saved plot: visualizations/feature_importance.png")

    # -----------------------------
    # Save Artifacts
    # -----------------------------
    joblib.dump(random_forest_model, os.path.join(models_dir, "fraud_detection_model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(models_dir, "feature_columns.pkl"))
    print("\nModel saved successfully into 'models/' folder!")

    # Sample Prediction Test
    sample_tx = df.drop("Class", axis=1).iloc[0].tolist()
    predict_transaction(sample_tx,
                        model_path=os.path.join(models_dir, "fraud_detection_model.pkl"),
                        scaler_path=os.path.join(models_dir, "scaler.pkl"),
                        feature_cols_path=os.path.join(models_dir, "feature_columns.pkl"))
    print("\n================ PIPELINE COMPLETED ================\n")


if __name__ == "__main__":
    run_pipeline()
