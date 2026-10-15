from flask import Flask, request, jsonify
import mlflow
import mlflow.pyfunc
from prometheus_client import Counter, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
import pandas as pd

app = Flask(__name__)

# MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load the registered model, version 1
# Load the trained MLflow model
MODEL_URI = (
    "mlruns/1/models/"
    "m-f63a1058c5bc458b836fcc3d703885df/artifacts"
)

model = mlflow.pyfunc.load_model(MODEL_URI)

# Prometheus metrics
PREDICTION_COUNT = Counter(
    "iris_predictions_total",
    "Total number of Iris predictions"
)

@app.route("/")
def home():
    return jsonify({
        "message": "Iris Classification API",
        "model": "IrisClassifier",
        "version": 1,
        "status": "running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "features" not in data:
        return jsonify({
            "error": "Provide four numeric values in the features field"
        }), 400

    features = data["features"]

    if not isinstance(features, list) or len(features) != 4:
        return jsonify({
            "error": "Exactly four features are required"
        }), 400

    try:
        features = [float(value) for value in features]
    except (TypeError, ValueError):
        return jsonify({
            "error": "All features must be numeric"
        }), 400

    # Iris feature order:
    # sepal length, sepal width, petal length, petal width
    columns = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]

    input_data = pd.DataFrame([features], columns=columns)
    prediction = int(model.predict(input_data)[0])

    species = [
        "setosa",
        "versicolor",
        "virginica"
    ]

    PREDICTION_COUNT.inc()

    return jsonify({
        "prediction": prediction,
        "species": species[prediction]
    })

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {
        "Content-Type": CONTENT_TYPE_LATEST
    }

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
