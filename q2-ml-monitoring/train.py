import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 3. Create an MLflow experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Iris Classification")

# 4. Start tracking the experiment
with mlflow.start_run() as run:

    # Define model parameters
    n_estimators = 100
    max_depth = 5

    # 5. Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    # 6. Train the model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 7. Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # 8. Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # 9. Save the trained model in MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        name="iris_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    # 10. Display results
    print("\nModel training completed successfully!")
    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Number of estimators: {n_estimators}")
    print(f"Maximum tree depth: {max_depth}")
    print(f"MLflow Run ID: {run.info.run_id}")
    print("Model saved successfully in MLflow.")
