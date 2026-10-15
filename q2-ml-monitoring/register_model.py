import mlflow
from mlflow import MlflowClient

# Connect to the local MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

# Find the Iris Classification experiment
experiment = client.get_experiment_by_name("Iris Classification")

if experiment is None:
    raise Exception("Iris Classification experiment not found")

# Find the latest successful run
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["attributes.start_time DESC"],
    max_results=1
)

if not runs:
    raise Exception("No successful training runs found")

run = runs[0]
run_id = run.info.run_id

print("Selected Run ID:", run_id)
print("Model Accuracy:", run.data.metrics.get("accuracy"))

# Register the logged model
model_uri = f"runs:/{run_id}/iris_model"

model_name = "IrisClassifier"

try:
    client.create_registered_model(model_name)
    print("Registered model created.")
except Exception:
    print("Registered model already exists.")

# Create a new model version
version = client.create_model_version(
    name=model_name,
    source=model_uri,
    run_id=run_id
)

print("Model registered successfully!")
print("Model Name:", model_name)
print("Model Version:", version.version)
print("Model Status:", version.status)
