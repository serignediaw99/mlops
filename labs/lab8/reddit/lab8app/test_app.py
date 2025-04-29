import requests
import json

def test_api():
    # Test the root endpoint
    response = requests.get('http://127.0.0.1:8001/')
    print("Root endpoint response:", response.json())
    
    # Test the prediction endpoint with sample iris data
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    response = requests.post(
        'http://127.0.0.1:8001/predict',
        json=test_data
    )
    print("\nPrediction endpoint response:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_api() 