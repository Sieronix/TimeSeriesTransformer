# transformer_model.py
# Defines the Transformer model with positional encoding and utility functions for training.

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Applies positional encoding to input tensors for sequence-based data.

    Args:
        d_model (int): Dimension of the model (embedding size).
        max_len (int): Maximum sequence length.
        frequency_scaling (float): Scaling factor for encoding frequencies, useful for wave-like data.
    """
    def __init__(self, d_model, max_len=5000, frequency_scaling=1.0):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        div_term *= frequency_scaling  # Adjust frequency scaling
        self.encoding[:, 0::2] = torch.sin(position * div_term)  # Apply sine for even indices
        self.encoding[:, 1::2] = torch.cos(position * div_term)  # Apply cosine for odd indices
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        """
        Adds positional encoding to input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Input tensor with positional encoding added.
        """
        seq_len = x.size(1)
        return self.encoding[:, :seq_len, :].to(x.device)


class TransformerModel(nn.Module):
    """
    Defines a Transformer model for sequence prediction tasks.

    Args:
        features_per_timestep (int): Number of features per timestep in input data.
        input_sequence_length (int): Length of input sequence.
        output_sequence_length (int): Length of output sequence.
        output_features (int): Number of features in output data. Default is 1.
        d_model (int): Dimension of the model (embedding size). Default is 64.
        nhead (int): Number of attention heads in the Transformer. Default is 8.
        num_layers (int): Number of Transformer encoder layers. Default is 2.
        dim_feedforward (int): Hidden layer size in feedforward network. Default is 256.
        dropout (float): Dropout probability. Default is 0.1.
        frequency_scaling (float): Scaling factor for positional encoding frequencies. Default is 1.0.
    """
    def __init__(self, features_per_timestep, input_sequence_length, output_sequence_length,
                 output_features=1, d_model=64, nhead=8, num_layers=2,
                 dim_feedforward=256, dropout=0.1, frequency_scaling=1.0):
        super().__init__()
        self.features_per_timestep = features_per_timestep
        self.output_sequence_length = output_sequence_length
        self.output_features = output_features

        # Embedding layer to project input features to the model dimension
        self.embedding = nn.Linear(features_per_timestep, d_model)

        # Positional encoding layer
        self.positional_encoding = PositionalEncoding(d_model, max_len=input_sequence_length, frequency_scaling=frequency_scaling)

        # Transformer encoder layers
        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True),
            num_layers=num_layers)

        # Sequence projection to map input sequence length to output sequence length
        self.sequence_projection = nn.Linear(input_sequence_length, output_sequence_length)

        # Final output layer to map to desired output features
        self.output_layer = nn.Linear(d_model, output_features)

    def forward(self, x):
        """
        Forward pass through the Transformer model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, features_per_timestep).

        Returns:
            Tensor: Predicted tensor of shape (batch_size, output_sequence_length, output_features).
        """
        # Apply embedding
        x = self.embedding(x)

        # Add positional encoding
        x += self.positional_encoding(x)

        # Pass through Transformer layers
        x = self.transformer_layers(x)

        # Project sequence length and apply the output layer
        x = x.transpose(1, 2)
        x = self.sequence_projection(x)
        x = x.transpose(1, 2)
        x = self.output_layer(x)

        return x


def create_model(features_per_timestep, input_sequence_length, output_sequence_length, frequency_scaling=1.0):
    """
    Factory function to create and initialize a Transformer model.

    Args:
        features_per_timestep (int): Number of features per timestep in input data.
        input_sequence_length (int): Length of input sequence.
        output_sequence_length (int): Length of output sequence.
        frequency_scaling (float): Scaling factor for positional encoding frequencies. Default is 1.0.

    Returns:
        TransformerModel: Initialized Transformer model.
    """
    set_seed(42)
    model = TransformerModel(
        features_per_timestep=features_per_timestep,
        input_sequence_length=input_sequence_length,
        output_sequence_length=output_sequence_length,
        frequency_scaling=frequency_scaling
    )
    return model


def set_seed(seed=42):
    """
    Sets the seed for reproducibility.

    Args:
        seed (int): Seed value. Default is 42.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
