import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
DATA_PATH, MODEL_PATH = 'processed_wisdm.npz', 'har_model.keras'
BATCH_SIZE, EPOCHS, DROPOUT_RATE = 32, 20, 0.3

def load_data(path):
    """Loads preprocessed WISDM data from npz file."""
    data = np.load(path, allow_pickle=True)
    return data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test']

def prepare_labels(y_train, y_val, y_test):
    """Encodes text labels to integers."""
    le = LabelEncoder(); return le.fit_transform(y_train), le.transform(y_val), le.transform(y_test), le

def build_cnn(input_shape, num_classes):
    """Defines a lightweight 1D-CNN architecture."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 3, activation='relu'),
        layers.Dropout(DROPOUT_RATE),
        layers.Conv1D(64, 3, activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def run_training(model, train_data, val_data):
    """Trains model with EarlyStopping."""
    XT, yT = train_data; XV, yV = val_data
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    return model.fit(XT, yT, validation_data=(XV, yV), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[es])

def report_metrics(model, sets, le):
    """Prints detailed accuracy and metrics for all splits."""
    for name, (X, y) in sets.items():
        loss, acc = model.evaluate(X, y, verbose=0)
        print(f"\n--- {name} Results ---\nAccuracy: {acc:.4f} | Loss: {loss:.4f}")
        if name == 'Test':
            y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
            print("\nClassification Report:\n", classification_report(y, y_pred, target_names=le.classes_))
            plot_cm(y, y_pred, le.classes_)

def plot_cm(y_true, y_pred, labels):
    """Saves confusion matrix heatmap."""
    plt.figure(figsize=(10, 8)); sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix'); plt.savefig('confusion_matrix.png'); plt.close()

def main():
    """Main execution flow for refined HAR model training."""
    # Data Loading
    print("Loading data..."); XT, yT, XV, yV, XTe, yTe = load_data(DATA_PATH)
    # Label Preparation
    yT_enc, yV_enc, yTe_enc, le = prepare_labels(yT, yV, yTe)
    # Model Construction & Training
    print("Building and training model with EarlyStopping..."); model = build_cnn(XT.shape[1:], len(le.classes_))
    run_training(model, (XT, yT_enc), (XV, yV_enc))
    # Detailed Reporting
    report_metrics(model, {'Train': (XT, yT_enc), 'Val': (XV, yV_enc), 'Test': (XTe, yTe_enc)}, le)
    # Model Saving
    model.save(MODEL_PATH); print(f"\nModel saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
