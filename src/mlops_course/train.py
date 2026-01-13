
import logging
import os

import hydra
import matplotlib.pyplot as plt
from typing import Annotated
import torch
import typer
from omegaconf import OmegaConf, DictConfig

from mlops_course.data import corrupt_mnist
from mlops_course.model import SimpleModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=f"{os.getcwd()}/configs", config_name="defaults.yaml")
def train(config: DictConfig) -> None:
    """Train a model on MNIST."""
    log.info(f"configuration: \n {OmegaConf.to_yaml(config)}")
    model_cfg = config.model
    train_cfg = config.training
    torch.manual_seed(train_cfg["seed"])

    model = SimpleModel(
        channels_in=model_cfg.channels_in,
        hidden_dims=model_cfg.hidden_dims,
        num_classes=model_cfg.num_classes,
        kernel_size=model_cfg.kernel_size,
        stride=model_cfg.stride,
        dropout_rate=model_cfg.dropout_rate,
    ).to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=train_cfg.batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    statistics: dict = {"train_loss": [], "train_accuracy": []}
    for epoch in range(train_cfg.epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                typer.echo(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    typer.echo("Training complete")
    os.makedirs(os.path.dirname(train_cfg.output_file), exist_ok=True)
    torch.save(model.state_dict(), train_cfg.output_file)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    os.makedirs("reports/figures", exist_ok=True)
    fig.savefig("reports/figures/training_statistics.png")


if __name__ == "__main__":
    train()
