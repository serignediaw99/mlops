import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

def load_data(data_path):
    # Load the dataset
    iris = pd.read_csv(data_path, header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
    return iris

def preprocess_data(df, test_size = 0.2, random_state = 42):
    # Separate features and target
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df[['species']]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)
    
    # Create and fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_processed_data(X_train, X_test, y_train, y_test, scaler, output_dir = 'data'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    # Save scaler
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

def main():
    # Load the data
    iris_df = load_data('data/iris.data')
    
    # Preprocess the data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(iris_df)
    
    # Save the processed data
    save_processed_data(X_train, X_test, y_train, y_test, scaler)

if __name__ == "__main__":
    main()
