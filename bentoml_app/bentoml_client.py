import bentoml
import numpy as np
from PIL import Image

if __name__ == "__main__":
    pil_img = Image.open("my_cat.jpg").convert("RGB")
    pil_img = pil_img.resize((224, 224))

    img_arr: np.ndarray = np.array(pil_img, dtype=np.float32)  # HWC
    img_arr = np.transpose(img_arr, (2, 0, 1))  # CHW
    img_arr = np.expand_dims(img_arr, axis=0)  # NCHW

    with bentoml.SyncHTTPClient("http://localhost:4040") as client:
        resp = client.predict(image=img_arr)
        print(resp)
