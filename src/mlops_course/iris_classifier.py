import pickle
from typing import Annotated, Any
from pathlib import Path

import typer
from sklearn.datasets import load_breast_cancer  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score, classification_report  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from sklearn.neighbors import KNeighborsClassifier  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
from sklearn.svm import SVC  # type: ignore[import-untyped]

app = typer.Typer()
train_app = typer.Typer()
app.add_typer(train_app, name="train")

# Load the dataset
data = load_breast_cancer()
x = data.data
y = data.target

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


@train_app.command()
def svm(kernel: str = "linear", output_file: Annotated[str, typer.Option("--output", "-o")] = "model.ckpt") -> None:
    """Train a SVM model."""
    model = SVC(kernel=kernel, random_state=42)
    model.fit(x_train, y_train)

    with open(output_file, "wb") as f:
        pickle.dump(model, f)


@train_app.command()
def knn(n_neighbors: int = 5, output_file: Annotated[str, typer.Option("--output", "-o")] = "model.ckpt") -> None:
    """Train a KNN model."""
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(x_train, y_train)

    with open(output_file, "wb") as f:
        pickle.dump(model, f)


@app.command()
def evaluate(model_file: Path) -> tuple[float, str]:
    """Evaluate the model."""
    with open(model_file, "rb") as f:
        model: Any = pickle.load(f)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    return accuracy, report


if __name__ == "__main__":
    app()
