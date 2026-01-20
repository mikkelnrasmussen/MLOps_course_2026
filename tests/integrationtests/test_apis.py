import os

import pytest
from fastapi.testclient import TestClient

import mlops_course.api as api_module
from mlops_course.api import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """
    Provide a TestClient while redirecting the API upload folder (FOLDER)
    to a temporary directory so tests don't touch /gcs/fastapi_app/.
    """
    monkeypatch.setattr(api_module, "FOLDER", str(tmp_path))
    return TestClient(app)


def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}


def test_read_item_valid_id(client):
    response = client.get("/items/123")
    assert response.status_code == 200
    assert response.json() == {"item_id": 123}


def test_read_item_invalid_id_returns_422(client):
    # FastAPI validates item_id as int; non-int should fail with 422
    response = client.get("/items/not-an-int")
    assert response.status_code == 422


def test_upload_file_saves_file_and_returns_location(client):
    filename = "hello.txt"
    content = b"hello from test"

    response = client.post(
        "/upload/",
        files={"file": (filename, content, "text/plain")},
    )

    assert response.status_code == 200
    body = response.json()
    assert "info" in body
    # Confirm response mentions both filename and the redirected folder
    assert filename in body["info"]

    # Verify file actually exists and contents match
    saved_path = os.path.join(api_module.FOLDER, filename)
    assert os.path.exists(saved_path)
    with open(saved_path, "rb") as f:
        assert f.read() == content


def test_list_files_includes_uploaded_file(client):
    # Arrange: create a file in the upload folder
    filename = "existing.bin"
    path = os.path.join(api_module.FOLDER, filename)
    with open(path, "wb") as f:
        f.write(b"\x00\x01")

    # Act
    response = client.get("/files/")

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert "files" in body
    assert filename in body["files"]
