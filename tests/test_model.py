import pytest
import torch

from mlops_course.model import SimpleModel


@pytest.mark.parametrize("batch_size", [32, 64, 128, 256])
def test_model(batch_size: int) -> None:
    model = SimpleModel()
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10)


def test_error_on_wrong_shape():
    model = SimpleModel()
    with pytest.raises(RuntimeError):
        model(torch.randn(1, 2, 3))
    with pytest.raises(RuntimeError):
        model(torch.randn(1, 1, 5, 5))
