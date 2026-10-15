import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

print("Registered model versions:")

for version in client.search_model_versions("name='IrisClassifier'"):
    print("-" * 40)
    print("Version:", version.version)
    print("Run ID:", version.run_id)
    print("Source:", version.source)
    print("Status:", version.status)
