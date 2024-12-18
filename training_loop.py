# training_loop.py
# Implements the training loop for the Transformer model, including loss tracking and model saving.

import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from transformer_model import create_model
import matplotlib.pyplot as plt

# Configure Matplotlib for use with interactive backends
plt.switch_backend('TkAgg')

# -----------------------------------------------------------------------------------------
# Configuration
save_path_for_trained_model = 'trained_models/trained_model.h5'  # Path to save the trained model
epochs = 10  # Number of training epochs
lr = 0.001  # Learning rate
# -----------------------------------------------------------------------------------------


def initialize_training_components(model):
    """
    Initializes the loss function, optimizer, and gradient scaler for training.

    Args:
        model (torch.nn.Module): The Transformer model.

    Returns:
        tuple: Loss criterion, optimizer, and gradient scaler.
    """
    criterion = nn.HuberLoss()  # Huber loss for regression tasks
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scaler = GradScaler()  # For mixed precision training
    return criterion, optimizer, scaler


def train_one_epoch(model, dataloader, criterion, optimizer, scaler):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The Transformer model.
        dataloader (DataLoader): DataLoader providing training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        scaler (GradScaler): Gradient scaler for mixed precision training.

    Returns:
        float: Average loss over the epoch.
    """
    model.train()
    total_loss = 0.0

    for batch_inputs, batch_targets in dataloader:
        optimizer.zero_grad()

        with autocast(device_type="cuda", dtype=torch.float16):
            predictions = model(batch_inputs)  # Forward pass
            loss = criterion(predictions, batch_targets)  # Compute loss

        scaler.scale(loss).backward()  # Backward pass with scaled loss
        scaler.step(optimizer)  # Update model parameters
        scaler.update()  # Update gradient scaler

        total_loss += loss.item()  # Accumulate total loss

    average_loss = total_loss / len(dataloader)
    return average_loss


def save_model_weights(model):
    """
    Saves the trained model's weights to a file.

    Args:
        model (torch.nn.Module): The trained model.
    """
    torch.save(model.state_dict(), save_path_for_trained_model)
    print(f"Model weights saved to '{save_path_for_trained_model}'.")


def graph_epoch_vs_loss(epoch_losses):
    """
    Plots the training loss over epochs.

    Args:
        epoch_losses (list): List of average losses for each epoch.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), epoch_losses, marker="o", label="Training Loss")
    plt.title("Epoch Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()


def training_loop_main(device, dataloader):
    """
    Main training loop for the Transformer model.

    Args:
        device (torch.device): Device for computation (CPU/GPU).
        dataloader (DataLoader): DataLoader providing training data.
    """
    # Initialize model
    inputs, targets = next(iter(dataloader))  # Get a sample batch to infer input/output dimensions
    model = create_model(features_per_timestep=inputs.size(2),
                         input_sequence_length=inputs.size(1),
                         output_sequence_length=targets.size(1))
    model = model.to(device)

    # Initialize training components
    criterion, optimizer, scaler = initialize_training_components(model)

    epoch_losses = []

    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        average_epoch_loss = train_one_epoch(model, dataloader, criterion, optimizer, scaler)
        epoch_losses.append(average_epoch_loss)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_epoch_loss:.5f}, Time: {epoch_time:.2f} seconds")

    # Save trained model and visualize training progress
    save_model_weights(model)
    graph_epoch_vs_loss(epoch_losses)
