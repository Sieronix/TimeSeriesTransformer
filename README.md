# **Waveformer: A Transformer for Time Series Analysis**

Waveformer is a specialized pipeline designed to handle **time-series data** using a **Transformer model**. This project focuses on leveraging historical data to create highly accurate predictions, making it ideal for wave-like patterns or any structured time-series data. The modular design allows for easy customization and scalability.

---

## **Key Features**

1. **Data Preprocessing**:
   - Cleans, selects, and scales raw time-series data.
   - Generates sliding window inputs for efficient transformer training.

2. **Transformer-Based Predictions**:
   - Utilizes a custom Transformer model with tailored positional encoding for wave-like or periodic data.
   - Supports multi-step forecasting with configurable input and output lengths.

3. **Integrated Training Pipeline**:
   - Supports mixed-precision training using PyTorch AMP for faster and memory-efficient model optimization.

4. **Evaluation and Visualization**:
   - Compares model predictions with actual values in both original and scaled formats.
   - Visualizes results with detailed plots for better interpretability.

5. **Modular and Extensible**:
   - Flexible design with dedicated modules for preprocessing, training, and evaluation.
   - Easily extendable for additional features or different types of time-series data.

---

## **System Architecture**

The project is divided into three main phases:

1. **Preprocessing**:
   - Loads raw data from pickle files.
   - Computes moving averages and creates scaled input-output windows.

2. **Training**:
   - Trains a Transformer model on the preprocessed data.
   - Saves trained models for reuse.

3. **Evaluation**:
   - Loads trained models and evaluates them on unseen samples.
   - Provides visual insights with input, target, and prediction comparisons.

---

## **Use Cases**

- Predicting financial market trends using historical price data.
- Forecasting demand in supply chain management.
- Analyzing sensor data for IoT applications.
- Weather prediction or climate modeling.

---
