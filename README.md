# WISDM Human Action Recognition (HAR)

A robust pipeline for classifying human activities from smartphone accelerometer data using the WISDM dataset.

## project Overview
This repository contains a complete end-to-end data science pipeline for HAR:
1.  **Exploratory Data Analysis**: Visualizing raw sensor signals and sampling intervals.
2.  **Preprocessing**: Cleaning malformed rows, 50Hz resampling, normalization, and window-based segmentation.
3.  **Modeling**: Comparison between a **1D-Convolutional Neural Network (DL)** and a **Random Forest (Baseline)**.
4.  **Inference**: A demo script to run predictions on processed data windows.

## Performance Benchmarks
| Model | Accuracy | F1-Score |
| :--- | :--- | :--- |
| **Random Forest (Best)** | **84.5%** | **0.78** |
| 1D-CNN | 71.9% | 0.72 |

## Key Features
- **Deterministic Preprocessing**: Fixed-rate resampling (50Hz) and overlap windowing (2s).
- **Subject-Wise Split**: Ensures the model generalizes to new users (Users 1-25 Train, 26-30 Val, 31-36 Test).
- **Inference Ready**: Trained models are saved in `.keras` and `.npz` formats.

## Installation & Usage

1. **Install Dependencies**:
```bash
pip install tensorflow scikit-learn matplotlib seaborn pandas scipy
```

2. **Run Pipeline**:
- `python3 raw_data_visualization.py`: Visualize raw data.
- `python3 preprocessing.py`: Process raw data into windows.
- `python3 har_model_training.py`: Train and evaluate the DL model.
- `python3 baseline_comparison.py`: Run the Random Forest baseline.
- `python3 inference_demo.py`: Test inference on a random sample.

## Analysis
The Random Forest model effectively handles the high similarity between vertical motion activities (Upstairs/Downstairs) better than the lightweight 1D-CNN. We found that class weighting in deep learning models can lead to a precision trade-off in dominant classes due to activity overlap in the WISDM dataset.

## License
MIT
