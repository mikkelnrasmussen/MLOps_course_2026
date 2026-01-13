"""Adapted from https://github.com/Jackson-Kang/PyTorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb.

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""

import logging
import os

import hydra
import torch
import torch.nn as nn
from mlops_course.vae_model import Decoder, Encoder, Model
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
import torchvision.transforms as transforms  # type: ignore[import-untyped]
from torchvision.datasets import MNIST  # type: ignore[import-untyped]
from torchvision.utils import save_image  # type: ignore[import-untyped]

log = logging.getLogger(__name__)

# creating a logger
@hydra.main(version_base=None, config_path=f"{os.getcwd()}/configs", config_name="defaults.yaml")
def train(config: DictConfig) -> None:
    """Train VAE on MNIST."""
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.model
    data_cfg = config.dataset
    torch.manual_seed(hparams["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data loading
    mnist_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = MNIST(data_cfg["dataset_path"], transform=mnist_transform, train=True, download=True)
    test_dataset = MNIST(data_cfg["dataset_path"], transform=mnist_transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=hparams["batch_size"], shuffle=False)

    encoder = Encoder(
        input_dim=hparams["x_dim"],
        hidden_dim=hparams["hidden_dim"],
        latent_dim=hparams["latent_dim"],
    )
    decoder = Decoder(
        latent_dim=hparams["latent_dim"],
        hidden_dim=hparams["hidden_dim"],
        output_dim=hparams["x_dim"],
    )

    model = Model(encoder=encoder, decoder=decoder).to(device)

    optimizer = hydra.utils.instantiate(config.optimizer, params=model.parameters(), lr=hparams["lr"])
    print(optimizer)

    def loss_function(x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + kld


    log.info("Start training VAE...")
    model.train()
    for epoch in range(hparams["n_epochs"]):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                print(batch_idx)
            x = x.view(x.size(0), hparams["x_dim"])
            x = x.to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()
        log.info(f"Epoch {epoch + 1} complete! Average Loss: {overall_loss / (batch_idx * hparams['batch_size'])}")
    log.info("Finish!!")

    # save weights
    os.makedirs(data_cfg["model_path"], exist_ok=True)
    torch.save(model, f"{data_cfg['model_path']}/trained_model.pt")

    # Generate reconstructions
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            if batch_idx % 100 == 0:
                print(batch_idx)
            x = x.view(x.size(0), hparams["x_dim"])
            x = x.to(device)
            x_hat, _, _ = model(x)
            break

    save_image(x.view(hparams["batch_size"], 1, 28, 28), f"{data_cfg['figure_path']}/orig_data.png")
    save_image(x_hat.view(hparams["batch_size"], 1, 28, 28), f"{data_cfg['figure_path']}/reconstructions.png")

    # Generate samples
    with torch.no_grad():
        noise = torch.randn(hparams["batch_size"], hparams["latent_dim"]).to(device)
        generated_images = decoder(noise)

    save_image(generated_images.view(hparams["batch_size"], 1, 28, 28), f"{data_cfg['figure_path']}/generated_sample.png")


if __name__ == "__main__":
    train()
