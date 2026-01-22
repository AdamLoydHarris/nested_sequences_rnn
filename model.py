"""
Neural network models for sequence working memory.

Implements GRU and LSTM with configurable hidden size.
Supports storing hidden states for analysis.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class SequenceGRU(nn.Module):
    """
    GRU-based sequence memory model.

    Stores hidden states at each timestep for representational analysis.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Storage for hidden states (for analysis)
        self.hidden_states: Optional[torch.Tensor] = None

    def forward(
        self,
        inputs: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        store_hidden: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            inputs: (batch, seq_len, input_dim)
            h0: Optional initial hidden state (n_layers, batch, hidden_dim)
            store_hidden: Whether to store hidden states for later analysis

        Returns:
            outputs: (batch, seq_len, output_dim) - logits
            h_final: (batch, hidden_dim) - final hidden state
        """
        if h0 is None:
            hidden_seq, h_n = self.gru(inputs)
        else:
            hidden_seq, h_n = self.gru(inputs, h0)

        outputs = self.output_layer(hidden_seq)

        if store_hidden:
            self.hidden_states = hidden_seq.detach()

        return outputs, h_n[-1]  # Return last layer's final state

    def get_hidden_states(self) -> Optional[torch.Tensor]:
        """Return stored hidden states from last forward pass."""
        return self.hidden_states

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state to zeros."""
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)


class SequenceLSTM(nn.Module):
    """
    LSTM-based sequence memory model.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.hidden_states: Optional[torch.Tensor] = None

    def forward(
        self,
        inputs: torch.Tensor,
        h0: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        store_hidden: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        if h0 is None:
            hidden_seq, (h_n, c_n) = self.lstm(inputs)
        else:
            hidden_seq, (h_n, c_n) = self.lstm(inputs, h0)

        outputs = self.output_layer(hidden_seq)

        if store_hidden:
            self.hidden_states = hidden_seq.detach()

        return outputs, h_n[-1]

    def get_hidden_states(self) -> Optional[torch.Tensor]:
        return self.hidden_states


def create_model(
    model_type: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    n_layers: int = 1,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: 'gru' or 'lstm'
        input_dim: Input dimension
        hidden_dim: Hidden state dimension
        output_dim: Output dimension
        n_layers: Number of recurrent layers
        dropout: Dropout rate between layers

    Returns:
        Initialized model
    """
    model_type = model_type.lower()

    if model_type == 'gru':
        return SequenceGRU(input_dim, hidden_dim, output_dim, n_layers, dropout)
    elif model_type == 'lstm':
        return SequenceLSTM(input_dim, hidden_dim, output_dim, n_layers, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation
    model = create_model('gru', input_dim=18, hidden_dim=256, output_dim=17)
    print(f"Model: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(32, 68, 18)  # batch=32, seq_len=68, input_dim=18
    y, h = model(x)
    print(f"Output shape: {y.shape}")
    print(f"Hidden shape: {h.shape}")
    print(f"Stored hidden shape: {model.get_hidden_states().shape}")
