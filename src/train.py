import yaml
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_model(name, params):
    if name == "logistic_regression":
        return LogisticRegression(**params)
    if name == "random_forest":
        return RandomForestClassifier(**params, random_state=42)
    if name == "svm":
        return SVC(**params)
    raise ValueError(f"Unknown model: {name}")


def main():
    config = load_config()
    exp_config = config["experiment"]
    data_config = config["data"]
    models_config = config["models"]

    mlflow.set_experiment(exp_config["name"])

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target,
        test_size=data_config["test_size"],
        random_state=exp_config["random_state"]
    )

    results = {}

    for model_name, params in models_config.items():
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_name", model_name)
            for k, v in params.items():
                mlflow.log_param(k, v)
            mlflow.log_param("test_size", data_config["test_size"])

            model = get_model(model_name, params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average="weighted")
            recall = recall_score(y_test, predictions, average="weighted")
            f1 = f1_score(y_test, predictions, average="weighted")

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Confusion matrix as an artifact image
            cm = confusion_matrix(y_test, predictions)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
            disp.plot()
            plt.title(f"Confusion Matrix - {model_name}")
            plot_path = f"reports/confusion_matrix_{model_name}.png"
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(plot_path)

            mlflow.sklearn.log_model(model, artifact_path="model")
            mlflow.log_artifact("configs/config.yaml")

            results[model_name] = accuracy
            print(f"{model_name}: accuracy={accuracy:.4f}, f1={f1:.4f}")

    best_model = max(results, key=results.get)
    print(f"\nBest model: {best_model} (accuracy={results[best_model]:.4f})")


if __name__ == "__main__":
    main()
