import requests
import numpy as np
import json

def test_api():
    """Sends a dummy sensor window to the Django backend for verification."""
    url = "http://127.0.0.1:8000/har/"
    # Create a dummy window (100 samples, 4 features)
    dummy_window = np.random.randn(100, 4).tolist()
    
    try:
        response = requests.post(url, json={'window': dummy_window})
        print(f"Status Code: {response.status_code}")
        print("Response Content:", json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_api()
