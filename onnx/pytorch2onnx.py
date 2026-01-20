import torch
import torchvision

model = torchvision.models.resnet18(weights=None)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "resnet18.onnx",
    input_names=["input.1"],
    dynamic_axes={"input.1": {0: "batch_size", 2: "height", 3: "width"}},
)
