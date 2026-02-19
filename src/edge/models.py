import torch
import torch.nn as nn
from typing import Tuple


class LSTMAnomalyDetector(nn.Module):
    """
    LSTM-based anomaly detector for multivariate time series.
    
    Uses encoder-decoder architecture to learn normal patterns
    and detect anomalies based on reconstruction error.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM anomaly detector.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMAnomalyDetector, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=hidden_size * self.num_directions,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, input_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            return_latent: Whether to return latent representation
        
        Returns:
            Tuple of (reconstruction, latent) if return_latent else (reconstruction,)
        """
        batch_size, seq_length, _ = x.size()
        
        # Encode
        encoder_output, (hidden, cell) = self.encoder(x)
        
        # Use the last encoder output as context
        if self.bidirectional:
            # Combine forward and backward hidden states
            hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
            hidden = hidden[:, 0, :, :] + hidden[:, 1, :, :]
            cell = cell.view(self.num_layers, 2, batch_size, self.hidden_size)
            cell = cell[:, 0, :, :] + cell[:, 1, :, :]
        
        # Decode: reconstruct the input sequence
        decoder_input = encoder_output
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
        
        # Generate reconstruction
        reconstruction = self.output_layer(decoder_output)
        
        if return_latent:
            # Return latent representation (last encoder hidden state)
            latent = hidden[-1]  # Shape: (batch_size, hidden_size)
            return reconstruction, latent
        
        return reconstruction
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            Latent representation of shape (batch_size, hidden_size)
        """
        _, (hidden, _) = self.encoder(x)
        
        if self.bidirectional:
            hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
            hidden = hidden[:, 0, :, :] + hidden[:, 1, :, :]
        
        return hidden[-1]
    
    def compute_reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute reconstruction error.
        
        Args:
            x: Input tensor
            reduction: How to reduce error ('mean', 'sum', or 'none')
        
        Returns:
            Reconstruction error
        """
        reconstruction = self.forward(x)
        error = torch.abs(x - reconstruction)
        
        if reduction == 'mean':
            return error.mean(dim=(1, 2))  # Mean over sequence and features
        elif reduction == 'sum':
            return error.sum(dim=(1, 2))
        else:
            return error


class AutoEncoder(nn.Module):
    """
    Autoencoder for detecting poisoned model updates in federated learning.
    """
    
    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 64,
        hidden_dims: list = None
    ):
        """
        Initialize autoencoder.
        
        Args:
            input_dim: Input dimension (flattened model parameters)
            encoding_dim: Encoding dimension
            hidden_dims: List of hidden layer dimensions
        """
        super(AutoEncoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = encoding_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor
        
        Returns:
            Tuple of (reconstruction, encoding)
        """
        encoding = self.encoder(x)
        reconstruction = self.decoder(encoding)
        return reconstruction, encoding
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        return self.decoder(z)
