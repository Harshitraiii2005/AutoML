import os
import yaml
import mlflow
from prefect import task, flow, get_run_logger
from datetime import timedelta
import pickle


from script.data_ingestion import load_dataset_from_mongo
from script.data_preprocessing import preprocess_and_split
from script.data_trainer import train_model
from script.s3_upload import upload_s3


class PickleModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)



@task(retries=2, retry_delay_seconds=5)
def load_config_task(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config



@task(retries=3, retry_delay_seconds=10)
def ingestion_task():
    logger = get_run_logger()
    logger.info("Starting data ingestion...")
    df = load_dataset_from_mongo()
    logger.info(f"Data ingestion completed. Shape: {df.shape}")
    return df



@task
def preprocessing_task(config):
    logger = get_run_logger()
    logger.info("Starting data preprocessing...")
    split_info = preprocess_and_split(config)
    logger.info("Data preprocessing completed")
    return split_info



@task(timeout_seconds=7200)
def training_task(config):
    logger = get_run_logger()
    logger.info("Starting model training...")

   
    if mlflow.active_run() is not None:
        logger.info(f"Ending previous active run: {mlflow.active_run().info.run_id}")
        mlflow.end_run()

    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])

   
    with mlflow.start_run(run_name="model_training") as run:
        model = train_model(config)  

        
        local_model_path = "model.pkl"
        with open(local_model_path, "wb") as f:
            pickle.dump(model, f)

        artifact_folder = "model"  
        
        mlflow.pyfunc.log_model(
            artifact_path=artifact_folder,
            python_model=PickleModelWrapper(model),
            artifacts={"model": local_model_path},
        )

        run_id = run.info.run_id

    logger.info(f"Model training completed. Run ID: {run_id}, Artifact Folder: {artifact_folder}")
    return run_id, artifact_folder



@task
def register_model_task(config, run_id, artifact_path="model"):
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    model_name = config["model_name"]  

    
    model_uri = f"runs:/{run_id}/{artifact_path}"

    mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"✅ Registered model '{model_name}' from run {run_id}")



@task
def s3_upload_task(config):
    logger = get_run_logger()
    if config.get("use_s3", False):
        logger.info("Uploading model to S3...")
        upload_s3(config)
        logger.info("Model uploaded to S3")
    else:
        logger.info("S3 upload skipped (use_s3=False)")



@flow(name="AutoML Training Pipeline")
def ml_pipeline():

    config = load_config_task()

    df = ingestion_task()
    preprocessing_task(config)

    run_id, artifact_folder = training_task(config)


    register_model_task(config, run_id, artifact_folder)

    s3_upload_task(config)




ml_pipeline()
