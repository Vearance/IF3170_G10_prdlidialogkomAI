"""Train a project classifier and predict rows from the cleaned test set."""

from __future__ import annotations

import importlib
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TRAIN_PATH = Path("data/clean/train.csv")
TEST_PATH = Path("data/clean/test.csv")
OUTPUT_DIR = Path("output")
TARGET = "is_canceled"
IGNORED_COLUMNS = ["id"]
ALGORITHMS = {
    "knn": "src.knn.alg:KNN",
    "logres": "src.logres.alg:LogisticRegression",
    "bayes": "src.bayes.alg:NaiveBayes",
    "dtl": "src.dtl.alg:DecisionTree",
    "svm": "src.svm.alg:SVM",
}


def _load_model(model_path: str) -> Any:
    """Load a model class specified as ``module.path:ClassName``."""
    try:
        module_path, class_name = model_path.split(":", maxsplit=1)
    except ValueError as err:
        raise ValueError("Model must use the format 'module.path:ClassName'.") from err

    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
    except (ImportError, AttributeError) as err:
        raise ValueError(f"Could not load model '{model_path}': {err}") from err

    if not callable(model_class):
        raise TypeError(f"'{model_path}' is not a callable model class.")
    return model_class()


def _features(data: pd.DataFrame, source: Path, *, training: bool) -> pd.DataFrame:
    if training and TARGET not in data.columns:
        raise ValueError(f"Target column '{TARGET}' was not found in '{source}'.")

    excluded = [column for column in [TARGET, *IGNORED_COLUMNS] if column in data.columns]
    features = data.drop(columns=excluded)
    non_numeric = features.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            f"'{source}' has non-numeric feature columns: {', '.join(non_numeric)}. "
            "Encode them during preprocessing first."
        )
    if features.isna().any().any():
        missing_columns = features.columns[features.isna().any()].tolist()
        raise ValueError(
            f"'{source}' has missing feature values in: {', '.join(missing_columns)}."
        )
    if features.shape[1] == 0:
        raise ValueError(f"No feature columns found in '{source}'.")
    return features


def run(model_path: str) -> np.ndarray:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    X_train = _features(train, TRAIN_PATH, training=True)
    X_test = _features(test, TEST_PATH, training=False)
    if train[TARGET].isna().any():
        raise ValueError(f"Target column '{TARGET}' in '{TRAIN_PATH}' contains missing values.")
    if X_train.columns.tolist() != X_test.columns.tolist():
        raise ValueError(
            "Train and test feature columns must match in the same order. "
            f"Train: {X_train.columns.tolist()}; test: {X_test.columns.tolist()}."
        )

    model = _load_model(model_path)
    model.fit(X_train.to_numpy(), train[TARGET].to_numpy())
    predictions = np.asarray(model.predict(X_test.to_numpy()))
    if predictions.shape != (len(test),):
        raise ValueError(
            "predict() must return one prediction per test row; "
            f"got shape {predictions.shape}, expected {(len(test),)}."
        )

    result = pd.DataFrame({TARGET: predictions})
    if "id" in test.columns:
        result.insert(0, "id", test["id"].to_numpy())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = OUTPUT_DIR / f"prediction_{timestamp}.csv"
    result.to_csv(output_path, index=False)

    print(f"Model: {model_path}")
    print(f"Training rows: {len(train)}")
    print(f"Test predictions: {len(predictions)}")
    print(f"Saved predictions to: {output_path}")
    return predictions


def _run_algorithm(name: str) -> None:
    try:
        run(ALGORITHMS[name])
    except (OSError, ValueError, TypeError) as err:
        raise SystemExit(f"Error running {name}: {err}") from err


def knn_main() -> None:
    _run_algorithm("knn")


def logres_main() -> None:
    _run_algorithm("logres")


def bayes_main() -> None:
    _run_algorithm("bayes")


def dtl_main() -> None:
    _run_algorithm("dtl")


def svm_main() -> None:
    _run_algorithm("svm")
