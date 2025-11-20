from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

# Feature names (defined once, reused everywhere)
features = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
    'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',
    'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
    'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

model = None
scaler = None

def load_artifacts():
    """Load model and scaler if available"""
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return True
    return False

loaded = load_artifacts()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict_page():
    return render_template("predict.html", features=features)

@app.route("/predict", methods=["POST"])
def predict():
    if not (model and scaler):
        return jsonify({"ok": False, "error": "Model artifacts not found. Please run train_model.py first to create model.pkl and scaler.pkl."}), 400

    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        ordered = []
        for key in features:
            if key not in data:
                return jsonify({"ok": False, "error": f"Missing feature: {key}"}), 400
            ordered.append(float(data[key]))

        X = np.array(ordered, dtype=float).reshape(1, -1)
        X_std = scaler.transform(X)

        pred = model.predict(X_std)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X_std)[0][1])

        return jsonify({
            "ok": True,
            "prediction": int(pred),
            "has_parkinsons": bool(pred == 1),
            "probability": proba
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
