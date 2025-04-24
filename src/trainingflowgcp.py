from metaflow import FlowSpec, step, Parameter, kubernetes, timeout, retry, catch, conda_base
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import os
import pickle

@conda_base(libraries={'scikit-learn':'1.2.2', 'pandas':'1.5.3', 'mlflow':'2.8.0', 'databricks-cli':'0.17.7'}, python='3.9.16')
class TrainingFlowGCP(FlowSpec):
    # Define parameters
    test_size = Parameter('test_size', default=0.2, type=float)
    random_state = Parameter('random_state', default=42, type=int)
    n_estimators = Parameter('n_estimators', default=100, type=int)
    mlflow_tracking_uri = Parameter('mlflow_tracking_uri', 
                                   default="https://mlflow-tracking-server-962158037709.us-west2.run.app", 
                                   help="MLflow tracking URI")
    # For GCP: mlflow_tracking_uri = Parameter('mlflow_tracking_uri', default="gs://your-bucket-name/mlflow", help="MLflow tracking URI")

    @retry(times=3)
    @kubernetes(cpu=0.5, memory=2000, image="python:3.9")
    @step
    def start(self):
        # Load Iris dataset
        iris_bunch = load_iris()
        self.X = iris_bunch['data']
        self.y = iris_bunch['target']
        self.feature_names = iris_bunch['feature_names']
        self.target_names = iris_bunch['target_names']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        print("Data loaded and split successfully")
        self.next(self.train)

    @retry(times=3)
    @timeout(minutes=10)
    @kubernetes(cpu=0.5, memory=2000, image="python:3.9")
    @step
    def train(self):
        # Initialize and train the model
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self.model.fit(self.X_train, self.y_train)
        
        # Save model to a file
        self.model_data = pickle.dumps(self.model)
        
        print(f"Model trained with {self.n_estimators} estimators")
        self.next(self.evaluate)

    @retry(times=3)
    @timeout(minutes=5)
    @catch(var='mlflow_error')
    @kubernetes(cpu=0.5, memory=2000, image="python:3.9")
    @step
    def evaluate(self):
        # Make predictions
        self.model = pickle.loads(self.model_data)
        y_pred = self.model.predict(self.X_test)
        
        # Calculate accuracy
        self.accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Model accuracy: {self.accuracy}")
        
        try:
            # Log to MLFlow
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            print(f"Using MLflow tracking URI: {self.mlflow_tracking_uri}")
            
            mlflow.set_experiment("iris-classification-gcp")
            
            with mlflow.start_run():
                # Log parameters
                mlflow.log_param("n_estimators", self.n_estimators)
                mlflow.log_param("test_size", self.test_size)
                mlflow.log_param("random_state", self.random_state)
                
                # Log metrics
                mlflow.log_metric("accuracy", self.accuracy)
                
                # Log model
                mlflow.sklearn.log_model(
                    self.model,
                    "iris-model-gcp",
                    registered_model_name="iris-classifier-gcp"
                )
                
                print("Successfully logged model to MLflow")
            
            if hasattr(self, 'mlflow_error') and self.mlflow_error:
                print(f"MLflow logging failed with error: {self.mlflow_error}")
                print("Continuing with flow execution despite MLflow error")
            
        except Exception as e:
            print(f"Error logging to MLflow: {e}")
            # Continue with the flow even if MLflow logging fails
            pass
        
        self.next(self.end)

    @retry(times=2)
    @step
    def end(self):
        print("Training flow completed successfully")
        print(f"Final model accuracy: {self.accuracy}")
        print("Model is registered in MLflow as 'iris-classifier-gcp'")

if __name__ == '__main__':
    TrainingFlowGCP() 