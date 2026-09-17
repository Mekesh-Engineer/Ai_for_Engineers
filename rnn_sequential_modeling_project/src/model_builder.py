# -*- coding: utf-8 -*-
"""
model_builder.py
----------------
ModelBuilderModule: Stacked Vanilla Recurrent Neural Network (RNN) Architecture
for Sequential Time-Series Regression Forecasting.
"""

import torch
import torch.nn as nn


class StackedRNNRegressor(nn.Module):
    """
    Stacked 2-Layer Vanilla Recurrent Neural Network (RNN) with Dropout Regularization
    and Linear Output for Sequential Time-Series Regression Forecasting.

    Architecture:
      - Input Sequence : (Batch, Timesteps=30, Input_Dim=1)
      - RNN Layer 1    : (Batch, Timesteps, 1) -> (Batch, Timesteps, Units_1=64) [batch_first=True, tanh]
      - Dropout 1      : Dropout(p=0.2)
      - RNN Layer 2    : (Batch, Timesteps, 64) -> (Batch, Timesteps, Units_2=32) [batch_first=True, tanh]
      - Dropout 2      : Dropout(p=0.2)
      - Last Hidden    : Extract timestep t=30 hidden representation (Batch, 32)
      - Linear Head    : Linear(32 -> 1) -> Predicted Next-Hour Energy Value (kWh)
    """

    def __init__(
        self,
        input_dim: int = 1,
        rnn_units_layer1: int = 64,
        rnn_units_layer2: int = 32,
        dropout_rate: float = 0.2,
        output_dim: int = 1,
    ):
        super(StackedRNNRegressor, self).__init__()

        self.input_dim = input_dim
        self.units1 = rnn_units_layer1
        self.units2 = rnn_units_layer2

        # Vanilla RNN Layer 1 (outputs full sequence: batch_first=True)
        self.rnn1 = nn.RNN(
            input_size=input_dim,
            hidden_size=rnn_units_layer1,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=True
        )
        self.dropout1 = nn.Dropout(p=dropout_rate)

        # Vanilla RNN Layer 2 (takes 64 features per timestep, outputs 32 hidden features)
        self.rnn2 = nn.RNN(
            input_size=rnn_units_layer1,
            hidden_size=rnn_units_layer2,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=True
        )
        self.dropout2 = nn.Dropout(p=dropout_rate)

        # Linear FC Output Layer
        self.fc = nn.Linear(rnn_units_layer2, output_dim)

        self._init_weights()

    def _init_weights(self):
        """Xavier / Orthogonal initialization for standard RNN and Linear layers."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0.0)
            elif "weight" in name and isinstance(param, nn.Linear):
                nn.init.xavier_uniform_(param.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through standard recurrent layers.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input sequence tensors of shape (Batch, Timesteps, Input_Dim).

        Returns
        -------
        torch.Tensor
            Predicted regression values of shape (Batch, Output_Dim).
        """
        # RNN Layer 1 -> returns all hidden states: (Batch, Timesteps, Units1)
        out1, _ = self.rnn1(x)
        out1 = self.dropout1(out1)

        # RNN Layer 2 -> returns all hidden states: (Batch, Timesteps, Units2)
        out2, _ = self.rnn2(out1)
        out2 = self.dropout2(out2)

        # Extract last timestep hidden state for sequence regression: (Batch, Units2)
        last_hidden = out2[:, -1, :]

        # Output linear projection
        out = self.fc(last_hidden)
        return out


def build_rnn_model(config: dict) -> StackedRNNRegressor:
    """Instantiate StackedRNNRegressor model using configuration parameters."""
    m_cfg = config.get("model", {})
    model = StackedRNNRegressor(
        input_dim=m_cfg.get("input_dim", 1),
        rnn_units_layer1=m_cfg.get("rnn_units_layer1", 64),
        rnn_units_layer2=m_cfg.get("rnn_units_layer2", 32),
        dropout_rate=m_cfg.get("dropout_rate", 0.2),
        output_dim=m_cfg.get("output_dim", 1),
    )
    return model


def summarize_model(model: nn.Module) -> None:
    """Print model architecture summary and parameter count."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n---------------- Stacked Vanilla RNN Model Architecture ----------------")
    print(model)
    print(f"  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {trainable_params:,}")
    print("------------------------------------------------------------------------\n")


if __name__ == "__main__":
    dummy_model = StackedRNNRegressor()
    summarize_model(dummy_model)
    dummy_input = torch.randn(32, 30, 1)  # (Batch=32, Timesteps=30, Features=1)
    dummy_output = dummy_model(dummy_input)
    print(f"[ModelBuilder Test] Forward pass test OK: Input {list(dummy_input.shape)} → Output {list(dummy_output.shape)}")
