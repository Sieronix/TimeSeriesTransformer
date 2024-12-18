# eval_model.py
# Evaluates a trained Transformer model by visualizing predictions against original and scaled values.

import torch
from transformer_model import TransformerModel
from preprocessing import input_length, output_length
import matplotlib.pyplot as plt

# Configure Matplotlib for interactive backend
plt.switch_backend('TkAgg')

# -----------------------------------------------------------------------------------------
# Configuration
load_path_for_transformer_model = 'trained_models/trained_model.h5'  # Path to load the trained model
max_input_points = 20  # Maximum number of input points to display in plots
# -----------------------------------------------------------------------------------------


def iterate_samples(processed_data):
    """
    Iterates through the processed data samples and adds an index for identification.

    Args:
        processed_data (list): List of processed data samples.

    Yields:
        dict: Dictionary containing sample data with an added "index" key.
    """
    for i, sample in enumerate(processed_data):
        yield {**sample, "index": i}


def unscale_prediction(prediction, mean, std):
    """
    Converts scaled predictions back to their original scale.

    Args:
        prediction (np.ndarray): Scaled prediction values.
        mean (float): Mean value used for scaling.
        std (float): Standard deviation used for scaling.

    Returns:
        np.ndarray: Unscaled prediction values.
    """
    return (prediction * std) + mean


def eval_transformer_main(device, processed_data):
    """
    Main evaluation function for the Transformer model.

    Args:
        device (torch.device): Device for computation (CPU/GPU).
        processed_data (list): List of processed data samples.
    """
    # Load the Transformer model and set to evaluation mode
    model = TransformerModel(1, input_length, output_length)
    model.load_state_dict(torch.load(load_path_for_transformer_model, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    print("Sample Amount:", len(processed_data))
    print("Input Length (per sample):", processed_data[0]["scaled_input"].shape[0])
    print("Target Length (per sample):", processed_data[0]["scaled_output"].shape[0])

    for sample in iterate_samples(processed_data):
        print(f"\nSample Index: {sample['index']}")
        for key, value in sample.items():
            if key != "index":
                print(f"{key.replace('_', ' ').capitalize()}: {value}")

        # Prepare input tensor for prediction
        single_loaded_input = torch.tensor(sample["scaled_input"], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        with torch.no_grad():
            single_loaded_prediction = model(single_loaded_input)

        # Convert predictions back to original scale
        scaled_prediction = single_loaded_prediction.squeeze().cpu().numpy()
        unscaled_prediction = unscale_prediction(scaled_prediction, sample["scaler"]["mean"], sample["scaler"]["std"])

        # Extract sample data for visualization
        original_input = sample["original_input"]
        original_output = sample["original_output"]
        scaled_input = sample["scaled_input"]
        scaled_output = sample["scaled_output"]

        # Plot results
        plot_results(
            sample_number=sample["index"],
            original_input=original_input,
            original_output=original_output,
            unscaled_prediction=unscaled_prediction,
            scaled_input=scaled_input,
            scaled_output=scaled_output,
            scaled_prediction=scaled_prediction
        )


def plot_results(sample_number, original_input, original_output, unscaled_prediction,
                 scaled_input, scaled_output, scaled_prediction):
    """
    Visualizes predictions and targets in both original and scaled forms.

    Args:
        sample_number (int): Index of the sample being visualized.
        original_input (np.ndarray): Original input data.
        original_output (np.ndarray): Original target data.
        unscaled_prediction (np.ndarray): Predictions in original scale.
        scaled_input (np.ndarray): Scaled input data.
        scaled_output (np.ndarray): Scaled target data.
        scaled_prediction (np.ndarray): Predictions in scaled form.
    """
    figsize = (12, 8)
    plt.figure(1, figsize=figsize)
    plt.clf()

    # Trim inputs for better visualization
    if max_input_points is not None and len(original_input) > max_input_points:
        trim_start = max(0, len(original_input) - max_input_points)
        original_input = original_input[trim_start:]
        scaled_input = scaled_input[trim_start:]

    # Adjust indices for plotting
    input_indices = range(len(original_input))
    target_indices = range(len(original_input), len(original_input) + len(original_output))

    # Top subplot: Original scale
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(input_indices, original_input, label="Original Input", color='blue', linestyle='-', marker='o', linewidth=2)
    ax1.plot(target_indices, original_output, label="Original Target", color='green', linestyle='--', marker='x', linewidth=2)
    ax1.plot(target_indices, unscaled_prediction, label="Unscaled Prediction", color='red', linestyle='-.', marker='^', linewidth=2)

    ax1.set_title(f"Sample Visualization (Original Scale) - Sample #{sample_number}")
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Values")
    ax1.grid(True)
    ax1.legend()

    # Bottom subplot: Scaled values
    ax2 = plt.subplot(2, 1, 2)
    scaled_input_indices = range(len(scaled_input))
    scaled_target_indices = range(len(scaled_input), len(scaled_input) + len(scaled_output))

    ax2.plot(scaled_input_indices, scaled_input, label="Scaled Input", color='blue', linestyle='-', marker='o', linewidth=2)
    ax2.plot(scaled_target_indices, scaled_output, label="Scaled Target", color='green', linestyle='--', marker='x', linewidth=2)
    ax2.plot(scaled_target_indices, scaled_prediction, label="Scaled Prediction", color='red', linestyle='-.', marker='^', linewidth=2)

    ax2.set_title(f"Sample Visualization (Scaled) - Sample #{sample_number}")
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Scaled Values")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.pause(0.1)
