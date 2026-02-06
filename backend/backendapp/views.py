import json
import numpy as np
import tensorflow as tf
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os

# Load model once on startup
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'har_model.keras')
MODEL = tf.keras.models.load_model(MODEL_PATH)
LABELS = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']

@csrf_exempt
def predict_activity(request):
    """API endpoint to predict human activity from accelerometer data."""
    if request.method != 'POST': return JsonResponse({'error': 'Only POST allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        # Expecting a window of shape (100, 4) or similar
        window = np.array(data['window']).astype('float32')
        if window.ndim == 2: window = np.expand_dims(window, axis=0)
        
        preds = MODEL.predict(window, verbose=0)
        idx = np.argmax(preds[0])
        
        return JsonResponse({
            'activity': LABELS[idx],
            'confidence': float(preds[0][idx]),
            'all_predictions': {LABELS[i]: float(preds[0][i]) for i in range(len(LABELS))}
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
