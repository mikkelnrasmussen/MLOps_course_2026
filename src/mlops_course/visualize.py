import matplotlib.pyplot as plt
import torch
import typer
import numpy as np
from numpy.typing import NDArray
from typing import List

from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.manifold import TSNE      # type: ignore[import-untyped]

from mlops_course.model import SimpleModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def visualize(model_checkpoint: str, figure_name: str = "embeddings.png") -> None:
    """Visualize model predictions."""
    model: torch.nn.Module = SimpleModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    model.fc1 = torch.nn.Identity()

    test_images: torch.Tensor = torch.load("data/processed/test_images.pt")
    test_target: torch.Tensor = torch.load("data/processed/test_target.pt")
    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)

    embedding_chunks: List[torch.Tensor] = []
    target_chunks: List[torch.Tensor] = []

    with torch.inference_mode():
        for images, target in torch.utils.data.DataLoader(test_dataset, batch_size=32):
            images = images.to(DEVICE)
            target = target.to(DEVICE)
            predictions: torch.Tensor = model(images)  # type: ignore[assignment]
            embedding_chunks.append(predictions.detach().cpu())
            target_chunks.append(target.detach().cpu())

    embeddings_np: NDArray[np.floating] = torch.cat(embedding_chunks).numpy()
    targets_np: NDArray[np.integer] = torch.cat(target_chunks).numpy()

    if embeddings_np.shape[1] > 500:
        pca = PCA(n_components=100)
        embeddings_np = pca.fit_transform(embeddings_np)

    tsne = TSNE(n_components=2)
    embeddings_2d: NDArray[np.floating] = tsne.fit_transform(embeddings_np)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        mask = targets_np == i
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=str(i))
    plt.legend()
    plt.savefig(f"reports/figures/{figure_name}")


if __name__ == "__main__":
    typer.run(visualize)