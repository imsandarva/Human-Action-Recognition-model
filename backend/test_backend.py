import requests
import numpy as np
import json

def test_api():
    """Sends a dummy sensor window to the Django backend for specialized inference."""
    url = "http://127.0.0.1:8000/har/"
    # Create a dummy window (100 samples, 3 features: x, y, z)
    samples = np.random.randn(100, 3).tolist()
    payload = {
        "sampling_rate": 50,
        "timestamp": 1675680000,
        "samples": samples
    }
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print("Response Content:", json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_api()
