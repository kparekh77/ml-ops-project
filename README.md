# End-to-End MLOps Pipeline with Drift Detection

This repository implements an automated pipeline that:

1. **Ingests** the UCI Adult dataset  
2. **Cleans & featurizes** it (train/test split, scaling, encoding)  
3. **Detects data drift** via KS‐tests  
4. **Conditionally retrains** a RandomForest model if drift is found  
5. **Logs** all experiments & models to MLflow  
6. **Orchestrates** the workflow daily with Airflow  

## Prerequisites

- Docker & Docker Compose (for local Airflow & MLflow)  
- `kind` (Kubernetes in Docker) if you want to deploy on a local cluster  
- Python 3.9+

## Quickstart

1. **Build the image**  
   ```bash
   docker build -t mlops-pipeline .

## Start and Run in Docker
    ```
    docker-compose down --volumes --remove-orphans   
    docker-compose build    
    docker-compose up -d 
    docker-compose run --rm webserver airflow db init 
    docker-compose run --rm webserver airflow users create \
     --username admin \
     --firstname Admin \
     --lastname User \
     --role Admin \
     --email admin@example.com \
     --password admin
    docker-compose up -d webserver scheduler 
    ```

Open localhost:8080 in your browser to access the Airflow UI, logging in with admin, admin and run the `mlops_pipeline` DAG.