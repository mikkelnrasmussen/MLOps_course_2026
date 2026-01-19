from __future__ import annotations

import re
from contextlib import asynccontextmanager
from enum import Enum
from http import HTTPStatus
from typing import Any

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Hello")
    yield
    print("Goodbye")


app = FastAPI(lifespan=lifespan)


class EmailRequest(BaseModel):
    email: str
    domain_match: str


@app.get("/")
def read_root() -> dict[str, str]:
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item_by_id(item_id: int) -> dict[str, int]:
    return {"item_id": item_id}


class ItemEnum(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


@app.get("/restrict_items/{item_id}")
def read_restricted_item(item_id: ItemEnum) -> dict[str, str]:
    return {"item_id": item_id}


@app.get("/query_items")
def read_query_item(item_id: int) -> dict[str, int]:
    return {"item_id": item_id}


database: dict[str, list[str]] = {"username": [], "password": []}


@app.post("/login/")
def login(username: str, password: str) -> str:
    username_db = database["username"]
    password_db = database["password"]
    if username not in username_db and password not in password_db:
        with open("database.csv", "a", encoding="utf-8") as file:
            file.write(f"{username}, {password}\n")
        username_db.append(username)
        password_db.append(password)
    return "login saved"


@app.post("/text_model/")
def contains_email(data: EmailRequest) -> dict[str, Any]:
    regex = r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+)\.[A-Z|a-z]{2,}\b"

    match = re.fullmatch(regex, data.email)
    is_email = match is not None

    domain_ok = False
    if match:  # mypy now knows match is not None
        domain = match.group(1)
        domain_ok = data.domain_match.lower() in domain.lower()

    return {
        "input": data.model_dump(),
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK.value,
        "is_email": is_email,
        "domain_match": domain_ok,
    }


@app.post("/cv_model/")
async def cv_model(
    data: UploadFile = File(...),
    h: int = 28,
    w: int = 28,
) -> FileResponse:
    with open("image.jpg", "wb") as image:
        content = await data.read()
        image.write(content)

    img = cv2.imread("image.jpg")
    if img is None:
        # simple guard; you could also raise HTTPException(400, ...)
        return FileResponse("image.jpg", status_code=HTTPStatus.BAD_REQUEST)

    res = cv2.resize(img, (w, h))  # (width, height)
    cv2.imwrite("image_resize.jpg", res)

    return FileResponse(
        "image_resize.jpg",
        status_code=HTTPStatus.OK,
        media_type="image/jpeg",
        filename="image_resize.jpg",
    )
