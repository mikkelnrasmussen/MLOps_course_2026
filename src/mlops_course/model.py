import torch
import torchmetrics
import torchvision  # type: ignore[import-untyped]
import wandb
from lightning import LightningModule
from torch import nn, optim


class SimpleModel(LightningModule):
    """Simple CNN model.

    Args:
        channels_in: Number of input channels.
        hidden_dims: List of hidden dimensions.
        num_classes: Number of output classes.
        kernel_size: Size of convolution kernel.
        stride: Stride of convolution.
        dropout_rate: Dropout rate.
        lr: Learning rate.
    """

    def __init__(
        self,
        channels_in: int = 1,
        hidden_dims: list = [32, 64, 128],
        num_classes: int = 10,
        kernel_size: int = 3,
        stride: int = 1,
        dropout_rate: float = 0.5,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.conv1 = nn.Conv2d(channels_in, hidden_dims[0], kernel_size, stride)
        self.conv2 = nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size, stride)
        self.conv3 = nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size, stride)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dims[2], num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor expected to be of shape [B, C, H, W].

        Returns:
            Output tensor of shape [B, num_classes].
        """
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)

    def evaluate(self, batch, stage=None):
        img, target = batch
        logits = self(img)
        loss = self.loss_fn(logits, target)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, target)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        """Training step."""
        if batch_idx % 200 == 0 and self.logger is not None and hasattr(self.logger, "experiment"):
            x, _ = batch
            grid = torchvision.utils.make_grid(x[:16].detach().cpu(), nrow=4, normalize=True)
            self.logger.experiment.log(
                {"train/samples": wandb.Image(grid, caption="Train batch samples")},
                step=int(self.global_step),
            )
        return self.evaluate(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")

    def configure_optimizers(self):
        """Configure optimizer."""
        return optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    model = SimpleModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
