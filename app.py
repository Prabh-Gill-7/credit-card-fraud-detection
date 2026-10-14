import os
import random
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(candidates):
    for c in candidates:
        full = os.path.join(BASE_DIR, c) if not os.path.isabs(c) else c
        if os.path.exists(full):
            return full
    return os.path.join(BASE_DIR, candidates[0])

# Locate model artifacts (support both subfolder and root structures)
MODEL_PATH = resolve_path([os.path.join("models", "fraud_detection_model.pkl"), "fraud_detection_model.pkl"])
SCALER_PATH = resolve_path([os.path.join("models", "scaler.pkl"), "scaler.pkl"])
FEATURES_PATH = resolve_path([os.path.join("models", "feature_columns.pkl"), "feature_columns.pkl"])
DATASET_PATH = resolve_path(["creditcard.csv", os.path.join("data", "creditcard.csv")])

model = None
scaler = None
feature_columns = []
dataset_df = None

def load_artifacts():
    global model, scaler, feature_columns, dataset_df
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURES_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_columns = joblib.load(FEATURES_PATH)
        print("[OK] Model artifacts loaded successfully.")
    else:
        print(f"[!] Artifacts missing at: {MODEL_PATH}")

    if os.path.exists(DATASET_PATH):
        dataset_df = pd.read_csv(DATASET_PATH)
        print(f"[OK] Dataset loaded with {len(dataset_df)} rows for sample generation.")

load_artifacts()

@app.route("/")
def index():
    return render_template("index.html", feature_columns=feature_columns)

@app.route("/plots/<path:filename>")
def serve_plot(filename):
    # Search in visualizations folder first, then root
    vis_dir = os.path.join(BASE_DIR, "visualizations")
    if os.path.exists(os.path.join(vis_dir, filename)):
        return send_from_directory(vis_dir, filename)
    return send_from_directory(BASE_DIR, filename)

@app.route("/api/sample/<sample_type>")
def get_sample(sample_type):
    global dataset_df
    if dataset_df is None or dataset_df.empty:
        sample = {f"V{i}": round(random.uniform(-1.5, 1.5), 4) for i in range(1, 29)}
        sample["Time"] = round(random.uniform(100, 150000), 2)
        sample["Amount"] = round(random.uniform(5, 500), 2)
        return jsonify({"success": True, "features": sample, "true_label": 0})

    if sample_type == "fraud":
        subset = dataset_df[dataset_df["Class"] == 1]
    elif sample_type == "legit":
        subset = dataset_df[dataset_df["Class"] == 0]
    else:
        subset = dataset_df

    if subset.empty:
        subset = dataset_df

    row = subset.sample(n=1).iloc[0]
    true_label = int(row["Class"])
    features = {col: float(row[col]) for col in feature_columns if col in row}

    return jsonify({
        "success": True,
        "features": features,
        "true_label": true_label,
        "true_label_text": "Fraudulent" if true_label == 1 else "Legitimate"
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"success": False, "error": "No input data provided"}), 400

        row_dict = {}
        for col in feature_columns:
            val = data.get(col, 0.0)
            try:
                row_dict[col] = [float(val)]
            except (ValueError, TypeError):
                row_dict[col] = [0.0]

        input_df = pd.DataFrame(row_dict)

        columns_to_scale = ["Time", "Amount"]
        input_df[columns_to_scale] = scaler.transform(input_df[columns_to_scale])

        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        if probability < 0.20:
            risk_level = "LOW RISK"
            badge_color = "green"
        elif probability < 0.60:
            risk_level = "MODERATE RISK"
            badge_color = "yellow"
        else:
            risk_level = "HIGH RISK"
            badge_color = "red"

        return jsonify({
            "success": True,
            "prediction": prediction,
            "label": "FRAUDULENT TRANSACTION" if prediction == 1 else "LEGITIMATE TRANSACTION",
            "is_fraud": prediction == 1,
            "probability": round(probability * 100, 2),
            "risk_level": risk_level,
            "badge_color": badge_color
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/batch-predict", methods=["POST"])
def batch_predict():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "Empty filename"}), 400

        df = pd.read_csv(file)
        if df.empty:
            return jsonify({"success": False, "error": "Uploaded CSV is empty"}), 400

        # Build feature DataFrame matching model requirements
        input_data = {}
        for col in feature_columns:
            if col in df.columns:
                input_data[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            else:
                input_data[col] = 0.0

        batch_df = pd.DataFrame(input_data)
        columns_to_scale = ["Time", "Amount"]
        batch_df[columns_to_scale] = scaler.transform(batch_df[columns_to_scale])

        predictions = model.predict(batch_df)
        probabilities = model.predict_proba(batch_df)[:, 1]

        total_count = len(df)
        fraud_count = int((predictions == 1).sum())
        legit_count = total_count - fraud_count
        fraud_rate = round((fraud_count / total_count) * 100, 2)

        # Flagged top fraudulent records (up to 20)
        flagged_indices = [i for i, p in enumerate(predictions) if p == 1][:20]
        flagged_list = []
        for idx in flagged_indices:
            row_dict = {
                "row_index": idx + 1,
                "amount": round(float(df["Amount"].iloc[idx]), 2) if "Amount" in df.columns else 0.0,
                "time": float(df["Time"].iloc[idx]) if "Time" in df.columns else 0.0,
                "probability": round(float(probabilities[idx]) * 100, 2)
            }
            flagged_list.append(row_dict)

        return jsonify({
            "success": True,
            "filename": file.filename,
            "total_count": total_count,
            "fraud_count": fraud_count,
            "legit_count": legit_count,
            "fraud_rate": fraud_rate,
            "flagged": flagged_list
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    print("\n" + "="*60)
    print(">> Fraud Detection Web Dashboard running on http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
