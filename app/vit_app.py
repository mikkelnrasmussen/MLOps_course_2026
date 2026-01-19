from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, cast

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from PIL.Image import Image as PILImage
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    print("Loading model")

    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    gen_kwargs: dict[str, int] = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}

    # Store on app.state (typed as Any), avoids "Name not defined" errors
    app.state.model = model
    app.state.feature_extractor = feature_extractor
    app.state.tokenizer = cast(PreTrainedTokenizerBase, tokenizer)
    app.state.device = device
    app.state.gen_kwargs = gen_kwargs

    yield

    print("Cleaning up")
    del app.state.model
    del app.state.feature_extractor
    del app.state.tokenizer
    del app.state.device
    del app.state.gen_kwargs


app = FastAPI(lifespan=lifespan)


@app.post("/caption/")
async def caption(data: UploadFile = File(...)) -> list[str]:
    """Generate a caption for an image."""
    i_image: PILImage = Image.open(data.file)
    if i_image.mode != "RGB":
        i_image = i_image.convert("RGB")

    # Pull from app.state
    model: VisionEncoderDecoderModel = app.state.model
    feature_extractor: ViTFeatureExtractor = app.state.feature_extractor
    tokenizer: PreTrainedTokenizerBase = app.state.tokenizer
    device: torch.device = app.state.device
    gen_kwargs: dict[str, Any] = app.state.gen_kwargs

    pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [pred.strip() for pred in preds]
