import numpy as np
import onnxruntime  # type: ignore[import-untyped]
from fastapi import FastAPI

app = FastAPI()


@app.get("/predict")
def predict():
    """Predict using ONNX model."""
    # Load the ONNX model
    model = onnxruntime.InferenceSession("resnet18.onnx")

    # Prepare the input data
    input_data = {"input.1": np.random.randn(1, 3, 224, 224).astype(np.float32)}

    # Run the model
    output = model.run(None, input_data)

    return {"output": output[0].tolist()}
