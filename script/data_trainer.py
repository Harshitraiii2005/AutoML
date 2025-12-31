import os
import yaml
import torch
import mlflow
import pandas as pd
import pickle
from packaging import version
import mlflow.pyfunc
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)


def load_config(path="config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


class QuestionGeneratorWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open("model.pkl", "rb") as f:
            self.model_bundle = pickle.load(f)

    def predict(self, context, model_input):
        return self.model_bundle["model"].generate(model_input)


class PandasDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def tokenize(tokenizer, df, config):
    if "input_text" not in df.columns or "target_text" not in df.columns:
        raise KeyError("Dataframe must contain 'input_text' and 'target_text' columns")

    inputs = ["Generate 5 questions: " + str(t) for t in df["input_text"]]
    targets = df["target_text"].astype(str).tolist()

    model_inputs = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=config.get("max_input_length", 512)
    )

    labels = tokenizer(
        targets,
        truncation=True,
        padding="max_length",
        max_length=config.get("max_target_length", 128)
    )

    model_inputs["labels"] = labels["input_ids"]
    return PandasDataset(model_inputs)


def train_model(config):
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["mlflow_experiment_name"])

    train_path = config.get("train_path", "dataset/train.csv")
    val_path = config.get("val_path", "dataset/validation.csv")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("Train or validation CSV not found")

    train_df = pd.read_csv(train_path).sample(n=2000, random_state=42)
    val_df = pd.read_csv(val_path).sample(n=2000, random_state=42)

    model_name = config.get("model_name", "t5-small")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    train_ds = tokenize(tokenizer, train_df, config)
    val_ds = tokenize(tokenizer, val_df, config)

    args = Seq2SeqTrainingArguments(
        output_dir=config.get("output_dir", "hf_model"),
        per_device_train_batch_size=config.get("batch_size", 1),
        per_device_eval_batch_size=config.get("batch_size", 1),
        num_train_epochs=config.get("num_epochs", 1),
        learning_rate=float(config.get("learning_rate", 5e-5)),
        logging_steps=50,
        save_steps=500,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer)
    )

    with mlflow.start_run(run_name="model_training", nested=True):
        trainer.train()

        model_dir = "hf_model"
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

        mlflow.log_artifacts(model_dir)

        with open("model.pkl", "wb") as f:
            pickle.dump({"model": model, "tokenizer": tokenizer}, f)

        mlflow.log_artifact("model.pkl")

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=QuestionGeneratorWrapper()
        )

    print("Training finished and PyFunc model logged")
    return "model"
