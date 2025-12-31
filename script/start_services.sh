#!/bin/bash
set -e

echo "Starting MLflow server on port 5090"
mlflow server \
  --host 0.0.0.0 \
  --port 5090 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  > mlflow.log 2>&1 &

sleep 5

echo "Starting Prefect server"
prefect server start \
  > prefect.log 2>&1 &

echo "Services started"
