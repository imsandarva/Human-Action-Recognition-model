import json, os, joblib, numpy as np, tensorflow as tf
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from scipy.interpolate import interp1d

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RF_PATH = os.path.join(BASE_DIR, 'rf_baseline_finetuned.joblib')
DL_PATH = os.path.join(BASE_DIR, 'har_model_finetuned.keras')
STD_PATH = os.path.join(BASE_DIR, 'global_std.json')
META_PATH = os.path.join(BASE_DIR, 'model_metadata.json')

# Global State
MODELS = {'rf': None, 'dl': None}
CONFIG = {'std': None, 'labels': None}
PREDICTIONS = [] # Last 50 predictions

def load_resources():
    """Load models and config once on startup."""
    global MODELS, CONFIG
    try:
        if os.path.exists(RF_PATH): MODELS['rf'] = joblib.load(RF_PATH)
        if os.path.exists(DL_PATH): MODELS['dl'] = tf.keras.models.load_model(DL_PATH)
        if os.path.exists(STD_PATH):
            with open(STD_PATH, 'r') as f: CONFIG['std'] = json.load(f)
        if os.path.exists(META_PATH):
            with open(META_PATH, 'r') as f: CONFIG['labels'] = json.load(f)['labels']
    except Exception as e: print(f"Startup Warning: {e}")

load_resources()

def preprocess_window(samples, sampling_rate):
    """Interpolate, mean-subtract, and scale the sensor window."""
    samples = np.array(samples)
    # Resample to 50Hz, 100 samples
    if len(samples) != 100 or sampling_rate != 50:
        t_old = np.linspace(0, len(samples)/sampling_rate, len(samples))
        t_new = np.linspace(0, 2.0, 100)
        samples = np.column_stack([interp1d(t_old, samples[:, i], kind='linear')(t_new) for i in range(3)])
    
    # Per-axis mean subtraction (window-wise)
    samples = samples - np.mean(samples, axis=0)
    
    # Scale by global standard deviation
    if CONFIG['std']:
        s_vec = np.array([CONFIG['std']['x'], CONFIG['std']['y'], CONFIG['std']['z']])
        samples = samples / s_vec
    
    # Compute magnitude
    mag = np.sqrt(np.sum(samples**2, axis=1)).reshape(-1, 1)
    return np.hstack([samples, mag])

@csrf_exempt
def predict_activity(request):
    """Highly specialized inference endpoint for HAR."""
    if request.method != 'POST': return JsonResponse({'error': 'POST required'}, status=405)
    try:
        data = json.loads(request.body)
        window = preprocess_window(data['samples'], data.get('sampling_rate', 50))
        
        # RF Prediction (Primary)
        rf_in = window.reshape(1, -1)
        rf_probs = MODELS['rf'].predict_proba(rf_in)[0]
        idx = np.argmax(rf_probs)
        label = CONFIG['labels'][idx] if CONFIG['labels'] else str(idx)
        
        res = {
            "label": label, "confidence": float(rf_probs[idx]), 
            "model": "rf", "timestamp": tf.timestamp().numpy().tolist()
        }
        
        # Optional DL Prediction
        if MODELS['dl']:
            dl_in = window.reshape(1, 100, 4)
            dl_pred = MODELS['dl'].predict(dl_in, verbose=0)[0]
            res["dl_result"] = {"label": CONFIG['labels'][np.argmax(dl_pred)], "confidence": float(np.max(dl_pred))}
            
        # Logging & State
        print(f"[{res['timestamp']}] Prediction: {res['label']} ({res['confidence']:.2f})")
        PREDICTIONS.append(res)
        if len(PREDICTIONS) > 50: PREDICTIONS.pop(0)
        
        return JsonResponse(res)
    except Exception as e:
        import traceback
        print(f"ERROR in /har/: {str(e)}")
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=400)

def get_latest(request):
    """Expose the last 50 predictions."""
    return JsonResponse({"latest": PREDICTIONS[-1] if PREDICTIONS else None, "history": PREDICTIONS[::-1]})

def dashboard(request):
    """Serve the dashboard page."""
    return render(request, 'dashboard.html')
