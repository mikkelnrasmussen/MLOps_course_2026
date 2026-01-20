import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Sequence, Tuple, cast

import anyio
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import models, transforms  # type: ignore[import-untyped]
from torchvision.transforms import Compose  # type: ignore[import-untyped]

# Typed globals for mypy
model: Optional[torch.nn.Module] = None
transform: Optional[Compose] = None
imagenet_classes: Optional[Sequence[str]] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global model, transform, imagenet_classes

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )

    labels_path = Path(__file__).with_name("imagenet-simple-labels.json")
    async with await anyio.open_file(labels_path, "r") as f:
        imagenet_classes = cast(Sequence[str], json.loads(await f.read()))

    yield

    model = None
    transform = None
    imagenet_classes = None


app = FastAPI(lifespan=lifespan)


def predict_image(image_path: str) -> Tuple[torch.Tensor, str]:
    """Predict image class given image path and return (probabilities, label)."""
    if model is None or transform is None or imagenet_classes is None:
        raise RuntimeError("Model not initialized. Lifespan startup did not run.")

    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_t)
    _, predicted_idx = torch.max(output, 1)

    probs = output.softmax(dim=-1)
    idx = int(predicted_idx.item())
    label = imagenet_classes[idx]
    return probs, label


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Hello from the backend!"}


@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        contents = await file.read()

        # file.filename can be None; avoid passing Optional into open_file/predict_image
        filename = file.filename or "upload.jpg"

        async with await anyio.open_file(filename, "wb") as f:
            await f.write(contents)

        probabilities, prediction = predict_image(filename)

        return {
            "filename": filename,
            "prediction": prediction,
            "probabilities": probabilities.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500) from e
