import pandas as pd
import numpy as np
import os, glob, json, joblib, time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf

# Paths and Constants
DATA_DIR, BACKEND_DIR, PLOT_DIR = 'collected_data', 'backend', 'plots'
MODEL_RF, MODEL_CNN = os.path.join(BACKEND_DIR, 'rf_baseline.joblib'), os.path.join(BACKEND_DIR, 'har_model.keras')
META_PATH, STD_PATH = os.path.join(BACKEND_DIR, 'model_metadata.json'), os.path.join(BACKEND_DIR, 'global_std.json')
SEED = 42; np.random.seed(SEED); tf.random.set_seed(SEED)

def setup():
    """Ensure directories and backups exist."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    for m in [MODEL_RF, MODEL_CNN]:
        if os.path.exists(m): joblib.dump(joblib.load(m) if m.endswith('joblib') else tf.keras.models.load_model(m), m + '.bak')

def load_and_preprocess():
    """Load, group, filter, and normalize collected data."""
    with open(META_PATH) as f: labels = json.load(f)['labels']
    with open(STD_PATH) as f: global_std = json.load(f)
    
    label_map = {l: i for i, l in enumerate(labels)}
    all_windows, all_labels, all_orig_files = [], [], []
    
    for f_path in glob.glob(os.path.join(DATA_DIR, "*.csv")):
        df = pd.read_csv(f_path)
        for wid, win in df.groupby('window_id'):
            if len(win) != 100: continue
            rate = win['avg_rate'].iloc[0]
            if rate < 35: continue # Drop low rate windows
            # Preprocess: subtract mean, divide by global std, compute magnitude
            arr = win[['ax', 'ay', 'az']].values
            arr = (arr - arr.mean(axis=0)) / [global_std['x'], global_std['y'], global_std['z']]
            mag = np.sqrt(np.sum(arr**2, axis=1)).reshape(-1, 1)
            all_windows.append(np.hstack([arr, mag]))
            all_labels.append(label_map[win['label'].iloc[0]])
            all_orig_files.append(os.path.basename(f_path))
            
    return np.array(all_windows), np.array(all_labels), np.array(all_orig_files), labels

def split_data(X, y, files):
    """80/10/10 chronological split per file."""
    train_idx, val_idx, test_idx = [], [], []
    for f in np.unique(files):
        idxs = np.where(files == f)[0]
        n = len(idxs); t, v = int(n * 0.8), int(n * 0.9)
        train_idx.extend(idxs[:t]); val_idx.extend(idxs[t:v]); test_idx.extend(idxs[v:])
    
    splits = {'train': (X[train_idx], y[train_idx]), 'val': (X[val_idx], y[val_idx]), 'test': (X[test_idx], y[test_idx])}
    with open('collected_splits.json', 'w') as f: json.dump({k: len(v[0]) for k, v in splits.items()}, f)
    return splits

def evaluate(model, X, y_true, model_name, labels):
    """Evaluate and save confusion matrix."""
    y_pred = rf_predict(model, X) if 'RF' in model_name else np.argmax(model.predict(X, verbose=0), axis=1)
    acc, f1 = accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix"); plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/collected_{model_name.lower().replace(' ', '_')}.png"); plt.close()
    return {'acc': acc, 'f1': f1, 'report': classification_report(y_true, y_pred, target_names=labels, output_dict=True)}

def rf_predict(rf, X): return rf.predict(X.reshape(len(X), -1))

def finetune_rf(splits, labels):
    """Retrain RF with original + new data."""
    orig = np.load('processed_wisdm.npz', allow_pickle=True)
    XT = np.vstack([orig['X_train'].reshape(len(orig['X_train']), -1), splits['train'][0].reshape(len(splits['train'][0]), -1)])
    yT = np.concatenate([orig['y_train'], [labels[i] for i in splits['train'][1]]])
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED).fit(XT, yT)
    # Map back to indices for evaluation consistency
    le_map = {l: i for i, l in enumerate(labels)}
    rf.classes_indexed = np.array([le_map[c] for c in rf.classes_])
    def predict_indexed(X): return rf.classes_indexed[rf.predict(X.reshape(len(X), -1)).astype(str) == rf.classes_[:, None]].flatten() # Simplified mapping
    # Overriding predict for indices
    class IndexedRF:
        def __init__(self, rf, labels): self.rf, self.labels, self.pmap = rf, labels, {l: i for i, l in enumerate(labels)}
        def predict(self, X): return np.array([self.pmap[c] for c in self.rf.predict(X)])
    irf = IndexedRF(rf, labels)
    joblib.dump(rf, 'rf_baseline_finetuned.joblib')
    return irf

def finetune_cnn(splits, num_classes):
    """Fine-tune CNN with low learning rate."""
    model = tf.keras.models.load_model(MODEL_CNN)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(splits['train'][0], splits['train'][1], validation_data=splits['val'], epochs=5, batch_size=16, verbose=0)
    model.save('har_model_finetuned.keras')
    return model

def plot_bar_chart(res_list, titles, labels):
    """Generate per-class F1 comparison bar chart."""
    x = np.arange(len(labels)); width = 0.2
    plt.figure(figsize=(12, 6))
    for i, (res, title) in enumerate(zip(res_list, titles)):
        f1s = [res['report'][l]['f1-score'] for l in labels]
        plt.bar(x + (i-1.5)*width, f1s, width, label=title)
    plt.xticks(x, labels, rotation=45); plt.ylabel('F1-Score'); plt.title('Per-Class F1 Improvement'); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/f1_comparison_bar.png", dpi=150); plt.close()

def main():
    setup(); print("Processing data..."); X, y, files, labels = load_and_preprocess()
    if len(X) < 10: print("Error: Insufficient windows collected (min 10 required). Aborting."); return
    splits = split_data(X, y, files); print(f"Data split: {len(X)} windows total.")
    
    print("Evaluating baselines..."); rf_base = joblib.load(MODEL_RF); cnn_base = tf.keras.models.load_model(MODEL_CNN)
    class BaseRFWrapper:
        def __init__(self, rf, labels): self.rf, self.map = rf, {l: i for i, l in enumerate(labels)}
        def predict(self, X): return np.array([self.map[p] for p in self.rf.predict(X.reshape(len(X), -1))])
    
    res_brf = evaluate(BaseRFWrapper(rf_base, labels), splits['test'][0], splits['test'][1], "Baseline RF", labels)
    res_bcnn = evaluate(cnn_base, splits['test'][0], splits['test'][1], "Baseline DL", labels)
    
    print("Fine-tuning Random Forest..."); ft_rf = finetune_rf(splits, labels)
    res_ftrf = evaluate(ft_rf, splits['test'][0], splits['test'][1], "Finetuned RF", labels)
    
    print("Fine-tuning CNN..."); ft_cnn = finetune_cnn(splits, len(labels))
    res_ftcnn = evaluate(ft_cnn, splits['test'][0], splits['test'][1], "Finetuned DL", labels)
    
    plot_bar_chart([res_brf, res_ftrf, res_bcnn, res_ftcnn], ["RF Base", "RF Fine", "DL Base", "DL Fine"], labels)
    
    # Save statistics for summary CSV and JSON
    summary_data = {
        'Metric': ['Accuracy', 'Macro-F1'],
        'RF Baseline': [res_brf['acc'], res_brf['f1']],
        'RF Finetuned': [res_ftrf['acc'], res_ftrf['f1']],
        'DL Baseline': [res_bcnn['acc'], res_bcnn['f1']],
        'DL Finetuned': [res_ftcnn['acc'], res_ftcnn['f1']]
    }
    pd.DataFrame(summary_data).to_csv('results_collected_summary.csv', index=False)
    
    with open('baseline_collected_results.json', 'w') as f:
        json.dump({'rf': {'acc': res_brf['acc'], 'f1': res_brf['f1']}, 'dl': {'acc': res_bcnn['acc'], 'f1': res_bcnn['f1']}}, f)
    
    # Update metadata
    if os.path.exists(META_PATH):
        with open(META_PATH, 'r+') as f:
            meta = json.load(f); meta['finetuned_on_user_data'] = True
            f.seek(0); json.dump(meta, f, indent=2); f.truncate()
    
    with open('REPORT_collected.md', 'w') as f:
        f.write("# HAR Fine-tuning Report\n\n## Comparison Metrics\n\n")
        f.write("| Model | Accuracy | Macro-F1 |\n| :--- | :--- | :--- |\n")
        f.write(f"| RF Baseline | {res_brf['acc']:.4f} | {res_brf['f1']:.4f} |\n")
        f.write(f"| RF Finetuned | {res_ftrf['acc']:.4f} | {res_ftrf['f1']:.4f} |\n")
        f.write(f"| DL Baseline | {res_bcnn['acc']:.4f} | {res_bcnn['f1']:.4f} |\n")
        f.write(f"| DL Finetuned | {res_ftcnn['acc']:.4f} | {res_ftcnn['f1']:.4f} |\n\n")
        f.write("## Visualizations\n- ![F1 Comparison](plots/f1_comparison_bar.png)\n- ![Finetuned RF CM](plots/collected_finetuned_rf.png)\n")
    
    print("\nSuccess! Results in REPORT_collected.md and plots/")

if __name__ == "__main__": main()
