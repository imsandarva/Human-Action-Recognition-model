import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_and_flatten(path):
    """Loads npz data and flattens windows for classic ML models."""
    d = np.load(path, allow_pickle=True)
    return d['X_train'].reshape(len(d['X_train']), -1), d['y_train'], d['X_test'].reshape(len(d['X_test']), -1), d['y_test']

def main():
    """Trains, evaluates, and saves the Random Forest baseline."""
    print("Loading data for baseline..."); XT, yT, XTe, yTe = load_and_flatten('processed_wisdm.npz')
    print("Training Random Forest baseline (this may take a moment)...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42).fit(XT, yT)
    y_pred = rf.predict(XTe)
    print(f"\n--- Baseline Random Forest Results ---\nAccuracy: {accuracy_score(yTe, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(yTe, y_pred))
    joblib.dump(rf, 'rf_baseline.joblib'); print("Model saved to rf_baseline.joblib")

if __name__ == "__main__":
    main()
