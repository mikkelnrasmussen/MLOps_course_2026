import torch
from torch import nn


class SimpleModel(nn.Module):
    """My awesome model."""

    def __init__(
            self, 
            channels_in: int = 1,
            hidden_dims: list = [32, 64, 128],
            num_classes: int = 10,
            kernel_size: int = 3,
            stride: int = 1,
            dropout_rate: float = 0.5,
        ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, hidden_dims[0], kernel_size, stride)
        self.conv2 = nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size, stride)
        self.conv3 = nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size, stride)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dims[2], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)


if __name__ == "__main__":
    model = SimpleModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
