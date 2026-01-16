import os
from typing import List

import typer
import wandb

app = typer.Typer()


@app.command()
def link_model(
    artifact_path: str = typer.Argument(...),
    alias: List[str] = typer.Option(["staging"], "--alias", "-a", help="Aliases to apply; repeatable."),
) -> None:
    """
    Stage a specific model to the model registry.

    Args:
        artifact_path: Path to the artifact to stage.
            Should be of the format "entity/project/artifact_name:version".
        aliases: List of aliases to link the artifact with.

    Example:
        model_management link-model entity/project/artifact_name:version -a staging -a best

    """
    if artifact_path == "":
        typer.echo("No artifact path provided. Exiting.")
        raise typer.Exit(code=0)

    api = wandb.Api(  # type: ignore[attr-defined]
        api_key=os.environ["WANDB_API_KEY"],
        overrides={"entity": os.environ["WANDB_ENTITY"], "project": os.environ["WANDB_PROJECT"]},
    )

    _, _, artifact_name_version = artifact_path.split("/")
    artifact_name, _ = artifact_name_version.split(":")

    artifact = api.artifact(artifact_path)

    REGISTRY_NAME = "Model"  # MUST match your Registry name in W&B UI
    COLLECTION_NAME = "corrupt_mnist_models"
    target_path = f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"

    artifact.link(
        target_path=f"{target_path}/{artifact_name}",
        aliases=alias,
    )
    artifact.save()
    typer.echo(f"Artifact {artifact_path} linked to {alias}")


if __name__ == "__main__":
    app()
