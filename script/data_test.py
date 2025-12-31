import json
import yaml
import torch
import mlflow
import pickle
import pandas as pd
import evaluate

from datasets import Dataset


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model(path, device):
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"].to(device)
    tokenizer = bundle["tokenizer"]
    model.eval()
    return model, tokenizer


def evaluate_model():
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["mlflow_experiment_name"])

    test_df = pd.read_csv("dataset/test.csv")
    df = test_df.sample(n=5000, random_state=42).reset_index(drop=True)
    dataset = Dataset.from_pandas(df)

    model, tokenizer = load_model("model.pkl", device)

    preds, refs = [], []

    for row in dataset.select(range(min(100, len(dataset)))):
        inputs = tokenizer(
            "Generate 5 questions: " + row["input_text"],
            return_tensors="pt",
            truncation=True
        ).to(device)

        outputs = model.generate(**inputs, max_length=128)
        preds.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        refs.append(row["target_text"])

    rouge = evaluate.load("rouge")
    metrics = rouge.compute(predictions=preds, references=refs)

    with mlflow.start_run(run_name="model_evaluation"):
        mlflow.log_metrics(metrics)

        with open("result.json", "w") as f:
            json.dump(metrics, f, indent=4)

        mlflow.log_artifact("result.json")
        print("📊 Evaluation:", metrics)


if __name__ == "__main__":
    evaluate_model()
