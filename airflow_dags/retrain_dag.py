from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 7, 16),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def _decide_branch():
    # read the flag file
    try:
        with open('/app/drift_flag.txt') as f:
            flag = f.read().strip()
    except FileNotFoundError:
        return 'train_model'  # if missing, just retrain
    return 'train_model' if flag == 'True' else 'end_pipeline'

with DAG(
    'mlops_retrain_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
) as dag:

    ingest = BashOperator(
        task_id='ingest_data',
        bash_command='python /app/scripts/ingest.py mode=ingest'
    )

    engineer = BashOperator(
        task_id='feature_engineering',
        bash_command='python /app/scripts/engineer.py mode=feature'
    )

    detect_drift = BashOperator(
        task_id='detect_drift',
        bash_command='python /app/scripts/drift.py mode=drift'
    )

    check_branch = BranchPythonOperator(
        task_id='branch_on_drift',
        python_callable=_decide_branch
    )

    train_model = BashOperator(
        task_id='train_model',
        bash_command='python /app/scripts/train.py mode=train'
    )

    end = EmptyOperator(task_id='end_pipeline')

    ingest >> engineer >> detect_drift >> check_branch
    check_branch >> [train_model, end]
