from fastapi import FastAPI
import uvicorn
import mlflow
import os
from pydantic import BaseModel
import numpy as np
from sklearn.datasets import load_iris

# Create FastAPI app
app = FastAPI(
    title="Iris Classifier",
    description="Classify iris flowers based on their features.",
    version="0.1",
)

# Define request body model
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Load the model at startup
@app.on_event("startup")
def load_model():
    global model, target_names
    try:
        # Set the MLflow tracking URI
        mlflow.set_tracking_uri("http://localhost:5000")
        print("MLflow tracking URI set successfully")
        
        # Load the iris classifier model
        model = mlflow.sklearn.load_model("models:/iris-classifier/latest")
        print("Model loaded successfully from MLflow")
        
        # Load iris dataset for target names
        iris = load_iris()
        target_names = iris.target_names
        print("Iris dataset loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e

# Root endpoint
@app.get('/')
def main():
    return {'message': 'This is a model for classifying iris flowers'}

# Prediction endpoint
@app.post('/predict')
def predict(iris_request: IrisRequest):
    # Convert input to numpy array
    features = np.array([
        iris_request.sepal_length,
        iris_request.sepal_width,
        iris_request.petal_length,
        iris_request.petal_width
    ]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)[0]
    prediction_proba = model.predict_proba(features)[0]
    
    return {
        'features': {
            'sepal_length': iris_request.sepal_length,
            'sepal_width': iris_request.sepal_width,
            'petal_length': iris_request.petal_length,
            'petal_width': iris_request.petal_width
        },
        'prediction': {
            'species': target_names[prediction],
            'probabilities': {
                target_names[i]: float(prob) for i, prob in enumerate(prediction_proba)
            }
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) 