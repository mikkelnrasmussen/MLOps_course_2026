import torch


def check_onnx_model(
    onnx_model_file: str,
    pytorch_model: torch.nn.Module,
    random_input: torch.Tensor,
    rtol: float = 1e-03,
    atol: float = 1e-05,
) -> None:
    import numpy as np
    import onnxruntime as rt  # type: ignore[import-untyped]

    ort_session = rt.InferenceSession(onnx_model_file)
    ort_inputs = {ort_session.get_inputs()[0].name: random_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    pytorch_outs = pytorch_model(random_input).detach().numpy()

    assert np.allclose(ort_outs[0], pytorch_outs, rtol=rtol, atol=atol)


if __name__ == "__main__":
    import torchvision  # type: ignore[import-untyped]

    model = torchvision.models.resnet18(weights=None)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        (dummy_input,),
        "resnet18.onnx",
        input_names=["input"],
        dynamic_axes={"input": {0: "batch_size", 2: "height", 3: "width"}},
    )

    check_onnx_model("resnet18.onnx", model, dummy_input)
