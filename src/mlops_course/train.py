import logging
import os

import hydra
import matplotlib.pyplot as plt
import torch
from hydra.core.hydra_config import HydraConfig  # type: ignore[import-untyped]
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as loguru_logger
from omegaconf import DictConfig, OmegaConf

import wandb
from mlops_course.data import corrupt_mnist
from mlops_course.model import SimpleModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=f"{os.getcwd()}/configs", config_name="defaults.yaml")
def train(config: DictConfig) -> None:
    hydra_path = HydraConfig.get().runtime.output_dir
    loguru_logger.add(os.path.join(hydra_path, "train.log"))

    log.info(f"configuration:\n{OmegaConf.to_yaml(config)}")
    model_cfg = config.model
    train_cfg = config.training

    seed_everything(train_cfg.seed, workers=True)

    # Data
    train_set, val_set = corrupt_mnist()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=train_cfg.batch_size, shuffle=False)

    # Model
    model = SimpleModel(
        channels_in=model_cfg.channels_in,
        hidden_dims=model_cfg.hidden_dims,
        num_classes=model_cfg.num_classes,
        kernel_size=model_cfg.kernel_size,
        stride=model_cfg.stride,
        dropout_rate=model_cfg.dropout_rate,
        lr=train_cfg.lr,
    )

    # W&B Logger (Lightning-managed run)
    resolved_cfg = OmegaConf.to_container(config, resolve=True)
    wandb_logger = WandbLogger(
        entity="minra-technical-university-of-denmark",
        project="MLOps_course",
        config=resolved_cfg,
        log_model="all",  # logs checkpoints as artifacts
        save_dir=hydra_path,
    )

    # Checkpointing
    ckpt_dir = os.path.join(hydra_path, "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")

    # Trainer device configuration (handles cuda/mps/cpu)
    if torch.cuda.is_available():
        accelerator = "cuda"
        devices = 1
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
    else:
        accelerator = "cpu"
        devices = 1

    trainer = Trainer(
        max_epochs=int(train_cfg.epochs),
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator=accelerator,
        devices=devices,
        default_root_dir=hydra_path,
        log_every_n_steps=10,
        limit_train_batches=0.2,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_ckpt = checkpoint_callback.best_model_path
    loguru_logger.info(f"Best checkpoint: {best_ckpt}")

    # Save a final model file (state_dict) and log as a W&B artifact
    os.makedirs(os.path.dirname(train_cfg.output_file), exist_ok=True)
    torch.save(model.state_dict(), train_cfg.output_file)

    # Log model artifact with final metrics, using the underlying wandb run
    run = wandb_logger.experiment
    artifact = wandb.Artifact(
        name="corrupt_mnist_models",
        type="model",
        description="A model trained to classify corrupt MNIST images",
    )
    artifact.add_file(best_ckpt)
    run.log_artifact(artifact)
    run.link_artifact(
        artifact=artifact,
        target_path="wandb-registry-Model/corrupt_mnist_models",
        aliases=["latest"],
    )

    loguru_logger.info("Training complete")


if __name__ == "__main__":
    train()
