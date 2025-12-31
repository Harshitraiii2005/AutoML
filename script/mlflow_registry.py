import mlflow
import yaml
import pickle
import os
import mlflow.pyfunc


class PickleModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input):
        return self.model.predict(model_input)


def load_config(path="config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def register_model():
    config = load_config()

    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config.get("mlflow_experiment_name", "registry"))

    model_path = "model.pkl"
    model_name = config["model_name"]

    with mlflow.start_run(run_name="register-pkl-pyfunc") as run:
        mlflow.pyfunc.log_model(
            artifact_path="model_artifact",
            python_model=PickleModel(),
            artifacts={"model": model_path}
        )
        run_id = run.info.run_id

    model_uri = f"runs:/{run_id}/model_artifact"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"Registered PyFunc model: {mv.name}, version: {mv.version}")
