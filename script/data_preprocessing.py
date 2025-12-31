import os
import yaml
import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5090")
mlflow.set_experiment("Auto-ML")


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def clean_text(s):
    if not isinstance(s, str):
        return ""
    return " ".join(s.replace("\n", " ").split())


def normalize_questions(q):
    if not isinstance(q, str):
        return ""
    parts = [p.strip() for p in q.replace("|", ".").split(".") if p.strip()]
    parts = parts[:5]
    return " ".join(f"{i+1}. {p}" for i, p in enumerate(parts))


def preprocess_and_split(config):
    with mlflow.start_run(run_name="data_preprocessing", nested=True):

        df = pd.read_csv("dataset/data.csv")
        print("Original shape:", df.shape)

        for col in ["paragraph", "summary", "key_points", "questions"]:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(clean_text)

        df["input_text"] = df["paragraph"]
        df["target_text"] = df["questions"].apply(normalize_questions)

        df = df[df["input_text"].str.len() > 20]
        df = df[df["target_text"].str.len() > 10]

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        n = len(df)
        train_end = int(config["train_split"] * n)
        val_end = train_end + int(config["val_split"] * n)

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        os.makedirs("dataset", exist_ok=True)

        train_path = "dataset/train.csv"
        val_path = "dataset/validation.csv"
        test_path = "dataset/test.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        mlflow.log_metrics({
            "train_rows": train_df.shape[0],
            "val_rows": val_df.shape[0],
            "test_rows": test_df.shape[0]
        })

        mlflow.log_artifacts("dataset", artifact_path="dataset")

        print("Preprocessing, splitting and saving completed")

    return {
        "train": {
            "path": train_path,
            "shape": train_df.shape
        },
        "val": {
            "path": val_path,
            "shape": val_df.shape
        },
        "test": {
            "path": test_path,
            "shape": test_df.shape
        }
    }
