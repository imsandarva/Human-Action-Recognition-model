import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import class_weight

# Constants
DATA_PATH, MODEL_PATH = 'processed_wisdm.npz', 'har_model.keras'
BATCH_SIZE, EPOCHS, DROPOUT_RATE = 32, 20, 0.3

def load_data(path):
    """Loads preprocessed WISDM data from npz file."""
    data = np.load(path, allow_pickle=True)
    return data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test']

def prepare_labels(y_train, y_val, y_test):
    """Encodes text labels and computes class weights."""
    le = LabelEncoder(); yT, yV, yTe = le.fit_transform(y_train), le.transform(y_val), le.transform(y_test)
    weights = dict(enumerate(class_weight.compute_class_weight('balanced', classes=np.unique(yT), y=yT)))
    return yT, yV, yTe, le, weights

def build_cnn(input_shape, num_classes):
    """Defines a lightweight 1D-CNN architecture."""
    model = models.Sequential([
        layers.Input(shape=input_shape), layers.Conv1D(64, 3, activation='relu'),
        layers.Dropout(DROPOUT_RATE), layers.Conv1D(64, 3, activation='relu'),
        layers.GlobalMaxPooling1D(), layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def run_training(model, train_data, val_data, weights):
    """Trains model with EarlyStopping and class weights."""
    XT, yT = train_data; XV, yV = val_data
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    return model.fit(XT, yT, validation_data=(XV, yV), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[es], class_weight=weights)

def report_metrics(model, sets, le):
    """Prints detailed accuracy and generates advanced visuals."""
    for name, (X, y) in sets.items():
        loss, acc = model.evaluate(X, y, verbose=0)
        print(f"\n--- {name} Results ---\nAccuracy: {acc:.4f} | Loss: {loss:.4f}")
        if name == 'Test':
            y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
            print("\nClassification Report:\n", classification_report(y, y_pred, target_names=le.classes_))
            plot_visuals(y, y_pred, le.classes_)

def plot_visuals(y_true, y_pred, labels):
    """Saves normalized confusion matrix and F1-score bar chart."""
    # Normalized Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title('Normalized Confusion Matrix'); plt.savefig('confusion_matrix_norm.png'); plt.close()
    # F1-Score Bar Chart
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    f1_scores = [report[label]['f1-score'] for label in labels]
    plt.figure(figsize=(10, 6)); plt.bar(labels, f1_scores, color='skyblue'); plt.ylabel('F1-Score'); plt.title('Per-Class F1-Score')
    plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.savefig('f1_scores_bar.png'); plt.close()

def main():
    """Main execution flow for refined HAR model training."""
    print("Loading data..."); XT, yT, XV, yV, XTe, yTe = load_data(DATA_PATH)
    yT_enc, yV_enc, yTe_enc, le, weights = prepare_labels(yT, yV, yTe)
    print("Building and training model with EarlyStopping & Class Weights..."); model = build_cnn(XT.shape[1:], len(le.classes_))
    run_training(model, (XT, yT_enc), (XV, yV_enc), weights)
    report_metrics(model, {'Train': (XT, yT_enc), 'Val': (XV, yV_enc), 'Test': (XTe, yTe_enc)}, le)
    model.save(MODEL_PATH); print(f"\nModel saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
