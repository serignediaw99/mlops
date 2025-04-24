from metaflow import FlowSpec, step, Parameter, Flow, kubernetes, timeout, retry, catch, conda_base
import mlflow
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

@conda_base(libraries={'scikit-learn':'1.2.2', 'pandas':'1.5.3', 'mlflow':'2.8.0', 'numpy':'1.23.5'}, python='3.9.16')
class ScoringFlowGCP(FlowSpec):
    # Define parameters
    input_data = Parameter('input_data', type=str, required=True)
    mlflow_tracking_uri = Parameter('mlflow_tracking_uri', default="http://localhost:5000", help="MLflow tracking URI")
    model_name = Parameter('model_name', default="iris-classifier-gcp", help="MLflow model name")

    @retry(times=3)
    @kubernetes
    @step
    def start(self):
        print(f"Starting scoring flow with input: {self.input_data}")
        
        # Parse input data 
        self.input_features = np.array([float(x) for x in self.input_data.split(',')]).reshape(1, -1)
        print(f"Input features shape: {self.input_features.shape}")
        
        # Load Iris dataset for reference
        iris_bunch = load_iris()
        self.feature_names = iris_bunch['feature_names']
        self.target_names = iris_bunch['target_names']
        
        self.next(self.load_model)

    @retry(times=3)
    @timeout(minutes=5)
    @catch(var='model_load_error')
    @kubernetes
    @step
    def load_model(self):
        try:
            # Load the latest trained model from MLFlow
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            print(f"Using MLflow tracking URI: {self.mlflow_tracking_uri}")
            
            # Load the model
            model_uri = f"models:/{self.model_name}/latest"
            print(f"Loading model from: {model_uri}")
            self.model = mlflow.sklearn.load_model(model_uri)
            print("Model loaded successfully from MLflow")
            
            if hasattr(self, 'model_load_error') and self.model_load_error:
                print(f"Model loading failed with error: {self.model_load_error}")
                self.model = None
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
        
        self.next(self.predict)

    @retry(times=2)
    @kubernetes
    @step
    def predict(self):
        if self.model is None:
            print("No model available for prediction. Skipping prediction step.")
            self.prediction = None
            self.prediction_proba = None
        else:
            try:
                # Make prediction
                print(f"Making prediction with input features shape: {self.input_features.shape}")
                self.prediction = self.model.predict(self.input_features)[0]
                self.prediction_proba = self.model.predict_proba(self.input_features)[0]
                print(f"Prediction made: {self.target_names[self.prediction]}")
            except Exception as e:
                print(f"Error making prediction: {e}")
                self.prediction = None
                self.prediction_proba = None
        
        self.next(self.end)

    @step
    def end(self):
        print("\nPrediction Results:")
        print(f"Input features: {dict(zip(self.feature_names, self.input_features[0]))}")
        
        if self.prediction is not None:
            print(f"Predicted class: {self.target_names[self.prediction]}")
            print("\nPrediction probabilities:")
            for i, prob in enumerate(self.prediction_proba):
                print(f"{self.target_names[i]}: {prob:.2f}")
        else:
            print("No prediction available due to errors in the flow.")

if __name__ == '__main__':
    ScoringFlowGCP() 