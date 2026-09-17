"""
model_builder.py
----------------
ModelBuilderModule: PyTorch Convolutional Neural Network (CNN) Architecture
for EEE Domain Solar PV Panel 4-Class Fault Classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_CNN(nn.Module):
    """
    Convolutional Neural Network Architecture for 28x28 Grayscale Thermography Image Classification.

    Architecture:
      - Block 1: Conv2D(1 -> 32, 3x3, pad=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2x2)  [28x28 -> 14x14]
      - Block 2: Conv2D(32 -> 64, 3x3, pad=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2x2) [14x14 -> 7x7]
      - Classifier: Flatten (64 * 7 * 7 = 3136) -> Linear(3136 -> 128) -> ReLU -> Dropout(0.5) -> Linear(128 -> num_classes)
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv1_filters: int = 32,
        conv2_filters: int = 64,
        fc1_units: int = 128,
        dropout_rate: float = 0.5,
        num_classes: int = 4,
    ):
        super(MNIST_CNN, self).__init__()

        # Conv Block 1
        self.conv1 = nn.Conv2d(in_channels, conv1_filters, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(conv1_filters)

        # Conv Block 2
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(conv2_filters)

        # Pooling & Dropout
        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Fully Connected Layers
        flatten_dim = conv2_filters * 7 * 7
        self.fc1 = nn.Linear(flatten_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Kaiming / He normal initialization for Conv and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input images of shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            Raw logits of shape (N, num_classes).
        """
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Classifier
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


# Alias for domain clarity
SolarPV_CNN = MNIST_CNN


def build_cnn_model(config: dict) -> MNIST_CNN:
    """Instantiate CNN model using configuration settings."""
    m_cfg = config.get("model", {})
    model = MNIST_CNN(
        in_channels=m_cfg.get("in_channels", 1),
        conv1_filters=m_cfg.get("conv1_filters", 32),
        conv2_filters=m_cfg.get("conv2_filters", 64),
        fc1_units=m_cfg.get("fc1_units", 128),
        dropout_rate=m_cfg.get("dropout_rate", 0.5),
        num_classes=m_cfg.get("num_classes", 4),
    )
    return model


def summarize_model(model: nn.Module) -> None:
    """Print CNN model architecture summary and parameter count."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n-- CNN Model Architecture (Solar PV Fault Detection) -------")
    print(model)
    print(f"  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {trainable_params:,}")
    print("-------------------------------------------------------------\n")


if __name__ == "__main__":
    dummy_model = MNIST_CNN(num_classes=4)
    summarize_model(dummy_model)
    dummy_input = torch.randn(4, 1, 28, 28)
    dummy_output = dummy_model(dummy_input)
    print(f"[ModelBuilder] Forward pass test OK: Input {list(dummy_input.shape)} -> Output {list(dummy_output.shape)}")
