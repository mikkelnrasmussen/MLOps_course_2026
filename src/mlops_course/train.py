import logging
import os
from typing import Annotated

import hydra
import matplotlib.pyplot as plt
from loguru import logger
import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score
import wandb

from mlops_course.data import corrupt_mnist
from mlops_course.model import SimpleModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=f"{os.getcwd()}/configs", config_name="defaults.yaml")
def train(config: DictConfig) -> None:
    """Train a model on MNIST."""
    # Get the path to the hydra output directory
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Add a log file to the logger
    logger.add(os.path.join(hydra_path, "train.log"))

    log.info(f"configuration: \n {OmegaConf.to_yaml(config)}")
    model_cfg = config.model
    train_cfg = config.training
    torch.manual_seed(train_cfg["seed"])

    resolved_cfg = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    run = wandb.init(
        project="MLOps_course",
        config=resolved_cfg
    )

    model = SimpleModel(
        channels_in=model_cfg.channels_in,
        hidden_dims=model_cfg.hidden_dims,
        num_classes=model_cfg.num_classes,
        kernel_size=model_cfg.kernel_size,
        stride=model_cfg.stride,
        dropout_rate=model_cfg.dropout_rate,
    ).to(DEVICE)
    train_set, val_set = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=train_cfg.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=train_cfg.batch_size, shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    statistics: dict = {"train_loss": [], "train_accuracy": []}
    for epoch in range(train_cfg.epochs):
        model.train()

        preds, targets = [], []
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
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())

            if i % 100 == 0:
                logger.info(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

                # add a plot of the input images
                grid = torchvision.utils.make_grid(img[:5].detach().cpu(), nrow=5, normalize=True)
                wandb.log({"images": wandb.Image(grid, caption="Input images")})

                # add a plot of histogram of the gradients
                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0).cpu()
                wandb.log({"gradients": wandb.Histogram(grads)})

        # add a custom matplotlib plot of the ROC curves
        preds = torch.cat(preds, 0)
        probs = torch.softmax(preds, dim=1)
        targets = torch.cat(targets, 0)

        fig, ax = plt.subplots(figsize=(8, 6))

        for class_id in range(10):
            one_hot = torch.zeros_like(targets)
            one_hot[targets == class_id] = 1

            RocCurveDisplay.from_predictions(
                one_hot.numpy(),                 # sklearn expects array-like
                probs[:, class_id].numpy(),
                name=f"class {class_id}",
                ax=ax,
            )

        ax.set_title("ROC Curves (one-vs-rest)")
        wandb.log({"roc": wandb.Image(fig)})

        # Evaluate on validation set
        model.eval()
        val_preds, val_targets = [], []
        val_loss = 0.0
        with torch.no_grad():
            for img, target in val_dataloader:
                img, target = img.to(DEVICE), target.to(DEVICE)
                y_pred = model(img)
                loss = loss_fn(y_pred, target)
                val_loss += loss.item() * img.size(0)
                val_preds.append(y_pred.detach().cpu())
                val_targets.append(target.detach().cpu())

        if len(val_preds) > 0:
            val_preds = torch.cat(val_preds, 0)
            val_probs = torch.softmax(val_preds, dim=1)
            val_targets = torch.cat(val_targets, 0)
            val_loss = val_loss / len(val_set)

            val_accuracy = accuracy_score(val_targets.numpy(), val_preds.argmax(dim=1).numpy())
            val_precision = precision_score(val_targets.numpy(), val_preds.argmax(dim=1).numpy(), average="weighted")
            val_recall = recall_score(val_targets.numpy(), val_preds.argmax(dim=1).numpy(), average="weighted")
            val_f1 = f1_score(val_targets.numpy(), val_preds.argmax(dim=1).numpy(), average="weighted")

            wandb.log({
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
            })

            logger.info(f"Epoch {epoch} validation - loss: {val_loss:.4f} acc: {val_accuracy:.4f} f1: {val_f1:.4f}")

            # plot validation ROC (one-vs-rest)
            fig_val, ax_val = plt.subplots(figsize=(8, 6))
            num_classes = model_cfg.num_classes if hasattr(model_cfg, "num_classes") else 10
            for class_id in range(num_classes):
                one_hot = torch.zeros_like(val_targets)
                one_hot[val_targets == class_id] = 1

                RocCurveDisplay.from_predictions(
                    one_hot.numpy(),
                    val_probs[:, class_id].numpy(),
                    name=f"class {class_id}",
                    ax=ax_val,
                )

            ax_val.set_title("Validation ROC Curves (one-vs-rest)")
            wandb.log({"val_roc": wandb.Image(fig_val)})

            plt.close(fig_val)
        
        plt.close(fig)

    final_accuracy = accuracy_score(targets, preds.argmax(dim=1))
    final_precision = precision_score(targets, preds.argmax(dim=1), average="weighted")
    final_recall = recall_score(targets, preds.argmax(dim=1), average="weighted")
    final_f1 = f1_score(targets, preds.argmax(dim=1), average="weighted")

    # first we save the model to a file then log it as an artifact
    torch.save(model.state_dict(), "model.pth")
    artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        description="A model trained to classify corrupt MNIST images",
        metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
    )
    artifact.add_file("model.pth")
    run.log_artifact(artifact)

    os.makedirs(os.path.dirname(train_cfg.output_file), exist_ok=True)
    torch.save(model.state_dict(), train_cfg.output_file)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    os.makedirs("reports/figures", exist_ok=True)
    fig.savefig("reports/figures/training_statistics.png")

    logger.info("Training complete")

if __name__ == "__main__":
    train()
