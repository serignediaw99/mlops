from metaflow import FlowSpec, step, Parameter, Flow
import mlflow
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

class ScoringFlow(FlowSpec):
    # Define parameters
    input_data = Parameter('input_data', type=str, required=True)

    @step
    def start(self):
        # Load the latest trained model from MLFlow
        mlflow.set_tracking_uri("http://localhost:5000")
        
        # Load the model
        self.model = mlflow.sklearn.load_model("models:/iris-classifier/latest")
        
        # Parse input data 
        self.input_features = np.array([float(x) for x in self.input_data.split(',')]).reshape(1, -1)
        
        # Load Iris dataset for reference
        iris_bunch = load_iris()
        self.feature_names = iris_bunch['feature_names']
        self.target_names = iris_bunch['target_names']
        
        self.next(self.predict)

    @step
    def predict(self):
        # Make prediction
        self.prediction = self.model.predict(self.input_features)[0]
        self.prediction_proba = self.model.predict_proba(self.input_features)[0]
        self.next(self.end)

    @step
    def end(self):
        print("\nPrediction Results:")
        print(f"Input features: {dict(zip(self.feature_names, self.input_features[0]))}")
        print(f"Predicted class: {self.target_names[self.prediction]}")
        print("\nPrediction probabilities:")
        for i, prob in enumerate(self.prediction_proba):
            print(f"{self.target_names[i]}: {prob:.2f}")

if __name__ == '__main__':
    ScoringFlow() 