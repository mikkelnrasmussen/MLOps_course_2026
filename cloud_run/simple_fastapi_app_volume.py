import os

from fastapi import FastAPI, File, UploadFile

app = FastAPI()


@app.get("/")
def read_root():
    """Root endpoint."""
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int):
    """Get an item by id."""
    return {"item_id": item_id}


FOLDER = "/gcs/fastapi_app/"


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(FOLDER, file.filename)
    with open(file_location, "wb") as f:
        contents = await file.read()
        f.write(contents)
    return {"info": f"file '{file.filename}' saved at '{file_location}'"}


@app.get("/files/")
def list_files():
    """List files in the upload folder."""
    files = os.listdir(FOLDER)
    return {"files": files}
