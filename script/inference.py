import os
import torch
import yaml
import argparse
import mlflow
import mlflow.pyfunc
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class QuestionGenerator(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        config = load_config()
        model_path = context.artifacts["model"]

        if config["use_lora"]:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                config["model_name"],
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.config = config

    def predict(self, context, model_input):
        if isinstance(model_input, str):
            texts = [model_input]
        elif hasattr(model_input, "iloc"):
            texts = model_input.iloc[:, 0].tolist()
        else:
            texts = model_input

        results = []
        for text in texts:
            prompt = f"Generate 5 high-quality questions: {text}"

            inputs = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.config["max_input_length"],
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.config["max_target_length"],
                    num_beams=4,
                    temperature=0.7
                )

            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            questions = [
                line.split(".", 1)[1].strip()
                for line in decoded.split("  ")
                if "." in line[:3]
            ]

            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "questions": questions
            })

        return results


def local_inference(text: str):
    config = load_config()

    if config["use_lora"]:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])
        model = PeftModel.from_pretrained(base_model, config["output_dir"])
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(config["output_dir"])
        tokenizer = AutoTokenizer.from_pretrained(config["output_dir"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    prompt = f"Generate 5 high-quality questions: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=config["max_target_length"])

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str)
    parser.add_argument("--log-model", action="store_true")
    args = parser.parse_args()

    config = load_config()
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["mlflow_experiment_name"])

    with mlflow.start_run(run_name="model_inference"):
        if args.log_model:
            mlflow.pyfunc.log_model(
                artifact_path="question_generator",
                python_model=QuestionGenerator(),
                artifacts={"model": config["output_dir"]},
            )
            print("MLflow PyFunc model logged")

        if args.text:
            local_inference(args.text)
