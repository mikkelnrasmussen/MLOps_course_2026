import os
import tempfile
import time

import pytest
import torch
import wandb

from mlops_course.model import SimpleModel


def load_model(model_checkpoint: str):
    api = wandb.Api(  # type: ignore[attr-defined]
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    artifact = api.artifact(model_checkpoint)

    logdir = tempfile.mkdtemp()
    artifact.download(root=logdir)
    file_name = artifact.files()[0].name
    return SimpleModel.load_from_checkpoint(f"{logdir}/{file_name}", weights_only=False)


@pytest.mark.performance
def test_model_speed():
    model = load_model(os.getenv("MODEL_NAME"))
    start = time.time()
    for _ in range(100):
        model(torch.rand(1, 1, 28, 28))
    end = time.time()
    assert end - start < 1
