FROM python:3.9-slim

# Install MLflow
RUN pip install mlflow

# Expose the default MLflow UI port
EXPOSE 5000

# Run MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
