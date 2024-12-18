# preprocessing.py
# Handles data preprocessing for Transformer model training and evaluation.

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from process_raw_data import process_raw_data_main
from src.utils.debug_utils import LOG_DEBUG_PRINT, TIME_FUNCTION

# -----------------------------------------------------------------------------------------
# Configuration
chosen_column_name = 'ma_16'  # Column name to use for the Transformer input
input_length = 100  # Length of input sequences
output_length = 15  # Length of output sequences
window_size = input_length + output_length  # Total window size for sliding window
preprocessing_batch_size = 10000  # Batch size for preprocessing
transformer_batch_size = 1028  # Batch size for Transformer DataLoader
# -----------------------------------------------------------------------------------------


def create_windows(data):
    """
    Creates sliding windows of data for training and testing.

    Args:
        data (pd.DataFrame): DataFrame containing the chosen column.

    Returns:
        np.ndarray: Array of sliding windows.
    """
    values = data[chosen_column_name].values
    cleaned_values = values[~np.isnan(values)]  # Remove NaN values
    return np.lib.stride_tricks.sliding_window_view(cleaned_values, window_size)


def custom_batch_scaler(batch_inputs, batch_outputs):
    """
    Scales each batch of inputs and outputs using mean and standard deviation.

    Args:
        batch_inputs (np.ndarray): Array of input sequences.
        batch_outputs (np.ndarray): Array of output sequences.

    Returns:
        tuple: Scaled inputs, scaled outputs, means, and standard deviations.
    """
    means = batch_inputs.mean(axis=1).reshape(-1, 1)
    stds = batch_inputs.std(axis=1, ddof=0).reshape(-1, 1)
    scaled_inputs = (batch_inputs - means) / stds
    scaled_outputs = (batch_outputs - means) / stds
    return scaled_inputs, scaled_outputs, means.flatten(), stds.flatten()


def process_samples_with_scaling(samples):
    """
    Processes sliding window samples and scales them for training.

    Args:
        samples (np.ndarray): Array of sliding window samples.

    Returns:
        list: List of dictionaries containing original and scaled inputs/outputs, and scalers.
    """
    structured_data = []

    for i in range(0, samples.shape[0], preprocessing_batch_size):
        batch = samples[i:i + preprocessing_batch_size]
        batch_inputs = batch[:, :input_length]
        batch_outputs = batch[:, input_length:]
        scaled_inputs, scaled_outputs, means, stds = custom_batch_scaler(batch_inputs, batch_outputs)
        structured_data.extend([
            {
                "original_input": batch_inputs[j],
                "original_output": batch_outputs[j],
                "scaled_input": scaled_inputs[j],
                "scaled_output": scaled_outputs[j],
                "scaler": {"mean": means[j], "std": stds[j]}
            }
            for j in range(batch.shape[0])
        ])
    return structured_data


def prepare_dataloader(processed_data, device):
    """
    Converts processed data into a PyTorch DataLoader for Transformer training.

    Args:
        processed_data (list): List of processed data dictionaries.
        device (torch.device): Device for tensor computation (CPU/GPU).

    Returns:
        DataLoader: PyTorch DataLoader with scaled inputs and outputs.
    """
    scaled_inputs = [sample["scaled_input"] for sample in processed_data]
    scaled_outputs = [sample["scaled_output"] for sample in processed_data]

    scaled_inputs = np.array(scaled_inputs, dtype=np.float32)
    scaled_outputs = np.array(scaled_outputs, dtype=np.float32)

    inputs = torch.tensor(scaled_inputs, dtype=torch.float32).unsqueeze(-1).to(device)
    targets = torch.tensor(scaled_outputs, dtype=torch.float32).unsqueeze(-1).to(device)

    LOG_DEBUG_PRINT("LOADED TENSOR DATA SHAPE", extra_lines=[
        f"Inputs Shape: {inputs.shape}",
        f"Targets Shape: {targets.shape}"
    ])

    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=transformer_batch_size, shuffle=True)

    return dataloader


def preprocessing_main(device):
    """
    Main function for data preprocessing.

    Args:
        device (torch.device): Device for tensor computation (CPU/GPU).

    Returns:
        tuple: DataLoader for training and processed data.
    """
    main_data = TIME_FUNCTION(process_raw_data_main)
    window_samples = TIME_FUNCTION(create_windows, main_data)
    processed_data = TIME_FUNCTION(process_samples_with_scaling, window_samples)
    dataloader = TIME_FUNCTION(prepare_dataloader, processed_data, device)
    return dataloader, processed_data
