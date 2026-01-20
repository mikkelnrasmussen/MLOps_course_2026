import json
import os
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
import requests  # type: ignore[import-untyped]
import streamlit as st
from google.cloud import run_v2


def get_backend_url() -> Optional[str]:
    parent = "projects/dtumlops-484510/locations/europe-west1"
    client = run_v2.ServicesClient()
    for service in client.list_services(parent=parent):
        if service.name.split("/")[-1] == "backend":
            return service.uri
    return os.environ.get("BACKEND", None)


def classify_image(image: bytes, backend: str) -> Optional[dict[str, Any]]:
    predict_url = f"{backend}/classify/"
    response = requests.post(predict_url, files={"file": image}, timeout=30)
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, dict):
            return cast(dict[str, Any], data)
    return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Image Classification")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = uploaded_file.read()
        result = classify_image(image, backend=backend)

        if result is not None:
            prediction = result["prediction"]
            probabilities = result["probabilities"]

            # show the image and prediction
            st.image(image, caption="Uploaded Image")
            st.write("Prediction:", prediction)

            # make a nice bar chart
            labels = json.loads(Path(__file__).with_name("imagenet-simple-labels.json").read_text())

            k = 10
            probs = np.array(probabilities, dtype=float).squeeze().ravel()  # force 1-D

            top_idx = np.argsort(probs)[-k:][::-1].tolist()  # list of Python ints

            df = pd.DataFrame(
                {
                    "Class": [labels[i] for i in top_idx],
                    "Probability": [float(probs[i]) for i in top_idx],
                }
            ).set_index("Class")

            st.bar_chart(df, y="Probability")
        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()
