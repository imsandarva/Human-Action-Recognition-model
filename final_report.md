# WISDM Human Action Recognition: Final Report

## Executive Summary
This project implemented a complete pipeline for Human Action Recognition (HAR) using the WISDM v1.1 dataset. We explored two primary modeling approaches: a **1D-CNN (Deep Learning)** and a **Random Forest (Classical ML)**. Surprisingly, the Random Forest model outperformed the deep learning approach in overall accuracy, which is a common phenomenon in sensor-based tasks where temporal features are well-captured by ensemble methods.

## Results Summary Table

| Model | Prep Stage | Sampling | Test Accuracy | F1 (Macro Avg) | Key Strength |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | Flattened | 2s Window | **0.8452** | **0.78** | Robust to noisy overlaps |
| **1D-CNN (Weights)** | CNN (1D) | 50 Hz | 0.7193 | 0.72 | High recall on minority classes |
| **1D-CNN (Nominal)** | CNN (1D) | 50 Hz | ~0.74* | 0.68 | Bias towards majority classes |

*\*Estimated based on initial training runs before class weighting.*

## Analysis of Model Performance

### Why Random Forest Won
1.  **Feature Locality**: Standard sensor data often has high variance. Random Forest's ability to create hard splits on specific sensor values (x, y, z) across a flattened window allowed it to distinguish between classes like 'Sitting' and 'Standing' with near-perfect precision (~1.00).
2.  **Handling Complexity**: The 1D-CNN architecture was kept lightweight for inference efficiency. While CNNs are great for temporal patterns, they can struggle with "Upstairs" vs. "Downstairs" overlap when the signals are visually similar and noisy.

### The Problem with Class Weights
In our DL experiment, adding class weights intended to balance the minority classes ('Upstairs', 'Downstairs') actually **hurt** overall accuracy.
- **Activity Overlap**: The sensor patterns for 'Upstairs' and 'Downstairs' are extremely similar. By "forcing" the model to pay more attention to them via weighting, it introduced more false positives for these classes, diluting the precision of 'Walking' and 'Jogging'.
- **Noise Sensitivity**: Minority classes in WISDM often contain more signal noise or ambiguous labeling. Weighting this noise amplified its impact on the gradients, leading to poorer convergence for the majority classes.

## Visual Evidence
Comparative visuals demonstrate that while the raw data was erratic, our cleaning pipeline successfully stabilized the sampling interval to a consistent 50Hz, providing a solid foundation for both models.

## Conclusion
For production deployment, the **Random Forest** model is recommended due to its superior accuracy (84.5%) and simpler interpretative nature for real-time sensor processing.
