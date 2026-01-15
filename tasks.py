import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "mlops_course"
PYTHON_VERSION = "3.12"


# General
@task
def python(ctx):
    """ """
    ctx.run("which python" if os.name != "nt" else "where python")


# Git commands
@task
def git(ctx: Context, message: str) -> None:
    """Commit and push changes to git."""
    ctx.run("git add .", echo=True, pty=not WINDOWS)
    ctx.run(f'git commit -m "{message}"', echo=True, pty=not WINDOWS)
    ctx.run("git push", echo=True, pty=not WINDOWS)


# DVC commands
@task
def dvc(ctx, folder="data", message="Add new data"):
    ctx.run(f"uvr dvc add {folder}")
    ctx.run(f"git add {folder}.dvc .gitignore")
    ctx.run(f"git commit -m '{message}'")
    ctx.run("git push")
    ctx.run("uvr dvc push")


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)


@task
def pull_data(ctx):
    ctx.run("uvr dvc pull")


# @task(pull_data)
# def train(ctx):
#     ctx.run("my_cli train")


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def train_vae(ctx: Context) -> None:
    """Train VAE model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/vae_mnist.py", echo=True, pty=not WINDOWS)


@task
def data_stats(ctx: Context) -> None:
    """Compute dataset statistics."""
    ctx.run(f"uv run src/{PROJECT_NAME}/dataset_statistics.py --datadir data/processed", echo=True, pty=not WINDOWS)


@task
def evaluate(ctx: Context) -> None:
    """Evaluate model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/evaluate.py models/model.pth", echo=True, pty=not WINDOWS)


@task
def visualize(ctx: Context) -> None:
    """Visualize model predictions."""
    ctx.run(f"uv run src/{PROJECT_NAME}/visualize.py models/model.pth", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
