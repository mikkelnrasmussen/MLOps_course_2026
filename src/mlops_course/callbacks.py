from __future__ import annotations

import matplotlib.pyplot as plt
import torch
import torchvision  # type: ignore[import-untyped]
from lightning.pytorch.callbacks import Callback
from sklearn.metrics import RocCurveDisplay  # type: ignore[import-untyped]

import wandb


class WandbExtrasCallback(Callback):
    def __init__(self, log_every_n_steps: int = 100, num_classes: int = 10) -> None:
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.num_classes = num_classes
        self._val_logits: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        x, y = batch
        x_cpu = x[:5].detach().cpu()
        grid = torchvision.utils.make_grid(x_cpu, nrow=5, normalize=True)

        # log via the active W&B experiment owned by WandbLogger
        if trainer.logger and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log(
                {"images": wandb.Image(grid, caption="Input images")}, step=trainer.global_step
            )

        # Gradient histogram (MPS/CUDA -> CPU first)
        grads = []
        for p in pl_module.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().flatten())
        if grads:
            g = torch.cat(grads).cpu().numpy().tolist()
            trainer.logger.experiment.log({"gradients": wandb.Histogram(g)}, step=trainer.global_step)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._val_logits.append(outputs["logits"].detach().cpu())
        self._val_targets.append(outputs["targets"].detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self._val_logits:
            return

        logits = torch.cat(self._val_logits, dim=0)
        targets = torch.cat(self._val_targets, dim=0)
        probs = torch.softmax(logits, dim=1)

        fig, ax = plt.subplots(figsize=(8, 6))
        for class_id in range(self.num_classes):
            one_hot = torch.zeros_like(targets)
            one_hot[targets == class_id] = 1

            RocCurveDisplay.from_predictions(
                one_hot.numpy(),
                probs[:, class_id].numpy(),
                name=f"class {class_id}",
                ax=ax,
            )

        ax.set_title("Validation ROC Curves (one-vs-rest)")

        if trainer.logger and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log({"val_roc": wandb.Image(fig)}, step=trainer.global_step)

        plt.close(fig)
        self._val_logits.clear()
        self._val_targets.clear()
