import mlflow

mlflow.set_experiment("MLOps-Lab-Demo")

with mlflow.start_run():
    mlflow.log_param("alpha", 0.5)
    mlflow.log_param("model_type", "dummy")
    mlflow.log_metric("accuracy", 0.87)
    mlflow.log_metric("loss", 0.23)

    with open("dummy_model.txt", "w") as f:
        f.write("this is a fake model artifact")
    mlflow.log_artifact("dummy_model.txt")

print("Run logged successfully!")
