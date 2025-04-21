from metaflow import FlowSpec, step, Parameter
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

class TrainingFlow(FlowSpec):
    # Define parameters
    test_size = Parameter('test_size', default=0.2, type=float)
    random_state = Parameter('random_state', default=42, type=int)
    n_estimators = Parameter('n_estimators', default=100, type=int)

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
        self.next(self.train)

    @step
    def train(self):
        # Initialize and train the model
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self.model.fit(self.X_train, self.y_train)
        self.next(self.evaluate)

    @step
    def evaluate(self):
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Log to MLFlow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("iris-classification")
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("n_estimators", self.n_estimators)
            mlflow.log_param("test_size", self.test_size)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            
            # Log model
            mlflow.sklearn.log_model(
                self.model,
                "iris-model",
                registered_model_name="iris-classifier"
            )
        
        self.next(self.end)

    @step
    def end(self):
        print("Training flow completed successfully")
        print(f"Model accuracy: {accuracy_score(self.y_test, self.model.predict(self.X_test))}")

if __name__ == '__main__':
    TrainingFlow() 