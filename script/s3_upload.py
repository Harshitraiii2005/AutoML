import boto3
import os
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def upload_s3(config: dict):
  
    if not config.get("use_s3", False):
        print("S3 upload skipped (use_s3=False)")
        return

    
    s3_config = config.get("s3", {})
    bucket = s3_config.get("bucket") or config.get("aws_bucket_name")
    s3_model_path = s3_config.get("model_path") or "model.pkl"
    region = s3_config.get("region") or config.get("aws_region")
    local_model_path = config.get("local_model_path", s3_model_path)

    if not bucket or not region:
        raise KeyError("S3 bucket or region not defined in config")

    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"Local model not found: {local_model_path}")

    
    s3 = boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=config.get("aws_access_key_id"),
        aws_secret_access_key=config.get("aws_secret_access_key")
    )

    
    s3.upload_file(
        Filename=local_model_path,
        Bucket=bucket,
        Key=s3_model_path
    )

    print(f"✅ {local_model_path} uploaded successfully to S3 bucket '{bucket}' at '{s3_model_path}'")
