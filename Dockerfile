# Dockerfile
FROM apache/airflow:2.5.0-python3.9

USER airflow

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir --user -r /requirements.txt

USER root
