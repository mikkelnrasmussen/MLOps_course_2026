import sys
import time
from statistics import mean, stdev

import onnxruntime as ort
import pytorch_lightning as pl
import torch
import torchvision
from torchvision.models import ResNet18_Weights


def timing_decorator(func, function_repeat: int = 10, timing_repeat: int = 5):
    """Decorator that times the execution of a function."""

    def wrapper(*args, **kwargs):
        timing_results = []
        for _ in range(timing_repeat):
            start_time = time.time()
            for _ in range(function_repeat):
                result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            timing_results.append(elapsed_time)
        print(f"{func.__name__}: Avg +- Stddev: {mean(timing_results):0.3f} +- {stdev(timing_results):0.3f} seconds")
        return result

    return wrapper


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.eval()

    def forward(self, x):
        return self.model(x)


model = LitModel().eval()


dummy_input = torch.randn(1, 3, 224, 224)
framework = "pytorch-ligtning"
if framework == "pytorch":
    torch.onnx.export(
        model,
        dummy_input,
        "resnet18.onnx",
        input_names=["input.1"],
        dynamic_axes={"input.1": {0: "batch_size", 2: "height", 3: "width"}},
    )
elif framework == "pytorch-ligtning":
    model.to_onnx(
        file_path="resnet18.onnx",
        input_sample=dummy_input,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input.1": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size"},
        },
    )


ort_session = ort.InferenceSession("resnet18.onnx")


@timing_decorator
def torch_predict(image) -> None:
    """Predict using PyTorch model."""
    model(image)


@timing_decorator
def onnx_predict(image) -> None:
    """Predict using ONNX model."""
    ort_session.run(None, {"input.1": image.numpy()})


if __name__ == "__main__":
    for size in [224, 448, 896]:
        dummy_input = torch.randn(1, 3, size, size)
        print(f"Image size: {size}")
        torch_predict(dummy_input)
        onnx_predict(dummy_input)
