import numpy as np
import tensorflow as tf

def main():
    """Loads model and runs inference on a single test sample."""
    print("Loading model and sample data..."); model = tf.keras.models.load_model('har_model.keras')
    data = np.load('processed_wisdm.npz', allow_pickle=True)
    X_test, y_test = data['X_test'], data['y_test']
    
    # Pick a random sample from test set
    idx = np.random.randint(0, len(X_test))
    sample, actual = X_test[idx:idx+1], y_test[idx]
    
    # Prediction
    pred_prob = model.predict(sample, verbose=0)
    # Re-encoding labels isn't strictly necessary for demo if we just want to show it works, 
    # but let's assume labels are known or use indices.
    print(f"\n--- Inference Demo ---\nSample Index: {idx}\nActual Activity: {actual}")
    print(f"Predicted Class Index: {np.argmax(pred_prob)} (Confidence: {np.max(pred_prob):.4f})")

if __name__ == "__main__":
    main()
