import json
import numpy as np
import tensorflow as tf
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os

from django.shortcuts import render

# Load model once on startup
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'har_model.keras')
MODEL = tf.keras.models.load_model(MODEL_PATH)
LABELS = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']

# Global state for dashboard updates
LATEST_PREDICTION = {'activity': 'Waiting...', 'confidence': 0.0, 'timestamp': None}

@csrf_exempt
def predict_activity(request):
    """API endpoint to predict human activity and update global state."""
    global LATEST_PREDICTION
    if request.method != 'POST': return JsonResponse({'error': 'POST only'}, status=405)
    try:
        data = json.loads(request.body)
        window = np.array(data['window']).astype('float32')
        if window.ndim == 2: window = np.expand_dims(window, axis=0)
        preds = MODEL.predict(window, verbose=0); idx = np.argmax(preds[0])
        LATEST_PREDICTION = {
            'activity': LABELS[idx],
            'confidence': float(preds[0][idx]),
            'timestamp': tf.timestamp().numpy()
        }
        return JsonResponse({'activity': LABELS[idx], 'confidence': float(preds[0][idx])})
    except Exception as e: return JsonResponse({'error': str(e)}, status=400)

def get_latest(request):
    """Returns the latest activity prediction for polling."""
    return JsonResponse(LATEST_PREDICTION)

def dashboard(request):
    """Renders the premium live dashboard."""
    return render(request, 'dashboard.html')
