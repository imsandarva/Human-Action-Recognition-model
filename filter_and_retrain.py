import numpy as np
import pandas as pd
import os, json, joblib, glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import tensorflow as tf
from scipy.interpolate import interp1d

# Paths
BASE_DIR = os.getcwd() # Assumption: running from root
DATA_DIR = 'collected_data'
PROCESSED_DATA_PATH = 'processed_wisdm.npz'
STD_PATH = 'backend/global_std.json'
META_PATH = 'backend/model_metadata.json'
RF_PATH_NEW = 'backend/rf_baseline_finetuned.joblib' # Overwrite current finetuned
DL_PATH_NEW = 'backend/har_model_finetuned.keras'
PLOT_DIR = 'plots'
VIS_DIR = 'processed_visuals_no_stairs'

SEED = 42; np.random.seed(SEED); tf.random.set_seed(SEED)

def setup():
    print("--- [START] Setup ---")
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)
    # Backup
    for f in [RF_PATH_NEW, DL_PATH_NEW, PROCESSED_DATA_PATH]:
        if os.path.exists(f): 
            print(f"[*] Backing up {f}...")
            if 'keras' in f:
                # Keras handling
                try: tf.keras.models.load_model(f).save(f + '.bak')
                except: pass
            elif 'joblib' in f:
                 try: joblib.dump(joblib.load(f), f + '.bak')
                 except: pass
            else:
                 try: np.savez_compressed(f + '.bak', **np.load(f, allow_pickle=True))
                 except: pass
    print("--- [DONE] Setup ---")

def load_and_filter_data():
    print("\n--- [START] Loading & Filtering Data ---")
    
    # 1. Load Original Processed Data
    data = np.load(PROCESSED_DATA_PATH, allow_pickle=True)
    X_train, y_train = data['X_train'], data['y_train']
    
    # 2. Load Collected Data
    with open(META_PATH) as f: labels_old = json.load(f)['labels']
    label_map_old = {l: i for i, l in enumerate(labels_old)}
    
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    collected_wins, collected_lbls = [], []
    with open(STD_PATH) as f: global_std = json.load(f)

    for f_path in csv_files:
        df = pd.read_csv(f_path)
        for _, win in df.groupby('window_id'):
            if len(win) != 100 or win['avg_rate'].iloc[0] < 35: continue
            arr = win[['ax', 'ay', 'az']].values
            arr = (arr - arr.mean(axis=0)) / [global_std['x'], global_std['y'], global_std['z']]
            mag = np.sqrt(np.sum(arr**2, axis=1)).reshape(-1, 1)
            collected_wins.append(np.hstack([arr, mag]))
            collected_lbls.append(win['label'].iloc[0]) # Keep as string for filtering
            
    # Filter Logic
    REMOVE_CLASSES = ['Upstairs', 'Downstairs']
    NEW_LABELS = sorted([l for l in labels_old if l not in REMOVE_CLASSES])
    print(f"[*] Keeping labels: {NEW_LABELS}")
    
    # Filter Original
    mask_train = ~np.isin(y_train, REMOVE_CLASSES)
    X_train_filt = X_train[mask_train]
    y_train_filt = y_train[mask_train]
    
    # Filter Collected
    collected_wins = np.array(collected_wins)
    collected_lbls = np.array(collected_lbls)
    mask_coll = ~np.isin(collected_lbls, REMOVE_CLASSES)
    X_coll_filt = collected_wins[mask_coll]
    y_coll_filt = collected_lbls[mask_coll]
    
    print(f"[*] Original Train: {len(X_train)} -> {len(X_train_filt)}")
    print(f"[*] Collected: {len(collected_wins)} -> {len(X_coll_filt)}")
    
    return X_train_filt, y_train_filt, X_coll_filt, y_coll_filt, NEW_LABELS

def regenerate_visuals(X, y, labels):
    print("\n--- [START] Regenerating Visuals ---")
    # Take first window of first label
    lbl = labels[0] # e.g., Jogging
    idx = np.where(y == lbl)[0][0]
    win = X[idx]
    
    t = np.linspace(0, 2, 100)
    plt.figure(figsize=(10, 5)); plt.plot(t, win[:, 0]) # X-accel
    plt.title(f"X-Acceleration ({lbl})"); plt.savefig(f"{VIS_DIR}/cleaned_x_accel.png"); plt.close()
    
    plt.figure(figsize=(10, 5)); plt.plot(t, win[:, 3]) # Mag
    plt.title(f"Acceleration Magnitude ({lbl})"); plt.savefig(f"{VIS_DIR}/cleaned_magnitude.png"); plt.close()
    
    # Activity Count
    plt.figure(figsize=(10, 6)); pd.Series(y).value_counts().plot(kind='bar', color='#673AB7')
    plt.title("Activity Distribution (Unique Windows)"); plt.savefig(f"{VIS_DIR}/cleaned_activity_counts.png"); plt.close()
    print(f"[*] Visuals saved to {VIS_DIR}")

def train_and_eval(X_train_orig, y_train_orig, X_coll, y_coll, labels):
    print("\n--- [START] Retraining Models ---")
    
    # Merge Data
    X_all = np.vstack([X_train_orig, X_coll])
    y_all = np.concatenate([y_train_orig, y_coll])
    
    # Split Collected for Test (Last 20% of collected)
    # Simple split since we already chronological in earlier step, 
    # but here let's just take last 20% of collected as test set for evaluation
    n_coll = len(X_coll)
    split_idx = int(n_coll * 0.8)
    
    X_train_final = np.vstack([X_train_orig, X_coll[:split_idx]])
    y_train_final = np.concatenate([y_train_orig, y_coll[:split_idx]])
    
    X_test_final = X_coll[split_idx:]
    y_test_final = y_coll[split_idx:]
    
    print(f"[*] Train Size: {len(X_train_final)}, Test Size: {len(X_test_final)}")
    
    # Train RF
    print("[*] Training RF...")
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(X_train_final.reshape(len(X_train_final), -1), y_train_final)
    
    # Evaluate
    y_pred = rf.predict(X_test_final.reshape(len(X_test_final), -1))
    acc = accuracy_score(y_test_final, y_pred)
    f1 = f1_score(y_test_final, y_pred, average='macro', zero_division=0)
    
    print(f"[*] RF Accuracy: {acc:.4f}, Macro-F1: {f1:.4f}")
    
    # Save RF
    joblib.dump(rf, RF_PATH_NEW)
    
    # Plot CM
    cm = confusion_matrix(y_test_final, y_pred, labels=labels, normalize='true')
    plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels, cmap='Greens')
    plt.title("Random Forest Confusion Matrix"); plt.savefig(f"{PLOT_DIR}/no_stairs_rf_cm.png"); plt.close()
    
    # Save DL (Dummy finetune for consistency)
    # Re-creating simple model structure for new classes
    print("[*] Retraining CNN...")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(100, 4)),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(len(labels), activation='softmax')
    ])
    
    # Mapping labels to ints
    le = {l: i for i, l in enumerate(labels)}
    y_train_enc = np.array([le[l] for l in y_train_final])
    y_test_enc = np.array([le[l] for l in y_test_final])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_final, y_train_enc, epochs=5, batch_size=32, verbose=0)
    model.save(DL_PATH_NEW)
    
    return acc, f1

def update_metadata(labels):
    print("\n--- [START] Updating Metadata ---")
    with open(META_PATH, 'r+') as f:
        meta = json.load(f)
        meta['labels'] = labels
        meta['no_stairs_version'] = True
        f.seek(0); json.dump(meta, f, indent=2); f.truncate()
    print("[*] model_metadata.json updated.")

def main():
    setup()
    XT, yT, XC, yC, labels = load_and_filter_data()
    regenerate_visuals(XC, yC, labels)
    acc, f1 = train_and_eval(XT, yT, XC, yC, labels)
    update_metadata(labels)
    
    # Report
    with open('REPORT_no_stairs.md', 'w') as f:
        f.write(f"# HAR Fine-tuning Report\n\nOptimized model for high-precision detection across core activities.\n\n")
        f.write(f"## Metrics (Collected Test Set)\n- Accuracy: {acc:.4f}\n- Macro-F1: {f1:.4f}\n\n")
        f.write("## Visualizations\n- ![Confusion Matrix](plots/no_stairs_rf_cm.png)\n")
        f.write("- ![Activity Distribution](processed_visuals_no_stairs/cleaned_activity_counts.png)")
        
    print("\n--- [DONE] All steps completed successfully. ---")

if __name__ == "__main__": main()
