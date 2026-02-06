import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
DATA_PATH = 'processed_wisdm.npz'
MODEL_PATH = 'har_model.keras'
BATCH_SIZE, EPOCHS = 32, 10

def load_data(path):
    """Loads preprocessed WISDM data from npz file."""
    data = np.load(path, allow_pickle=True)
    return data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test']

def prepare_labels(y_train, y_val, y_test):
    """Encodes text labels to integers."""
    le = LabelEncoder()
    return le.fit_transform(y_train), le.transform(y_val), le.transform(y_test), le

def build_cnn(input_shape, num_classes):
    """Defines a lightweight 1D-CNN architecture."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 3, activation='relu'),
        layers.Dropout(0.2),
        layers.Conv1D(64, 3, activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def run_training(model, train_data, val_data):
    """Trains the model with the given data."""
    X_train, y_train = train_data
    X_val, y_val = val_data
    return model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS)

def plot_confusion_matrix(y_true, y_pred, labels):
    """Generates and saves a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png'); plt.close()

def perform_evaluation(model, X_test, y_test, le):
    """Evaluates the model and reports metrics."""
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nAccuracy:", model.evaluate(X_test, y_test, verbose=0)[1])
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
    plot_confusion_matrix(y_test, y_pred, le.classes_)

def save_output(model, path):
    """Saves the trained model."""
    model.save(path); print(f"Model saved to {path}")

def main():
    """Main execution flow for HAR model training."""
    # Step 1: Data acquisition
    XT, yT, XV, yV, XTe, yTe = load_data(DATA_PATH)
    # Step 2: Label preparation
    yT_enc, yV_enc, yTe_enc, le = prepare_labels(yT, yV, yTe)
    # Step 3: Model construction
    model = build_cnn(XT.shape[1:], len(le.classes_))
    # Step 4: Training execution
    run_training(model, (XT, yT_enc), (XV, yV_enc))
    # Step 5: Evaluation & Reporting
    perform_evaluation(model, XTe, yTe_enc, le)
    # Step 6: Artifact persistence
    save_output(model, MODEL_PATH)

if __name__ == "__main__":
    main()
