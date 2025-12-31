import mlflow
import pandas as pd
import os
from pymongo import MongoClient

MONGO_URI = "add mongo url"
DB_NAME = "automl_db"
COLLECTION_NAME = "dataset"

RAW_PATH = "dataset/data.csv"

mlflow.set_tracking_uri("http://localhost:5090")
mlflow.set_experiment("Auto-ML")

def load_dataset_from_mongo():
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    data = list(collection.find())
    df = pd.DataFrame(data)

    if "_id" in df.columns:
        df.drop(columns=["_id"], inplace=True)

    df.to_csv(RAW_PATH, index=False)
    mlflow.log_artifact(RAW_PATH, artifact_path="dataset")
    print(f"Ingested dataset: {df.shape}")

    return df

if __name__ == "__main__":
    with mlflow.start_run(run_name="data_ingestion"):
        load_dataset_from_mongo()
