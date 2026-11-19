"""Evaluate a classifier on validation data and create a test submission."""

from __future__ import annotations

import argparse
import importlib
import itertools
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TRAIN_PATH = Path("data/clean/train.csv")
COMBINED_PATH = Path("data/clean/combined.csv")
VALIDATION_PATH = Path("data/clean/val.csv")
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


def _load_model(model_path: str, **kwargs: Any) -> Any:
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
    return model_class(**kwargs)


def _features(data: pd.DataFrame, source: Path, *, require_target: bool) -> pd.DataFrame:
    if require_target and TARGET not in data.columns:
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


def _with_spinner[Result](label: str, operation: Callable[[], Result]) -> Result:
    stop = threading.Event()

    def animate() -> None:
        symbols = itertools.cycle("|/-\\")
        while not stop.wait(0.1):
            sys.stderr.write(f"\r{label}... {next(symbols)}")
            sys.stderr.flush()

    sys.stderr.write(f"{label}... ")
    sys.stderr.flush()
    started_at = time.perf_counter()
    spinner = threading.Thread(target=animate, daemon=True)
    spinner.start()
    try:
        result = operation()
    except BaseException:
        stop.set()
        spinner.join()
        sys.stderr.write(f"\r{label}... failed\n")
        sys.stderr.flush()
        raise

    stop.set()
    spinner.join()
    elapsed = time.perf_counter() - started_at
    sys.stderr.write(f"\r{label}... done ({elapsed:.1f}s)\n")
    sys.stderr.flush()
    return result


def _macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = dict.fromkeys(np.concatenate((y_true, y_pred)).tolist())
    class_scores = []

    for label in labels:
        true_positive = np.sum((y_true == label) & (y_pred == label))
        false_positive = np.sum((y_true != label) & (y_pred == label))
        false_negative = np.sum((y_true == label) & (y_pred != label))
        denominator = 2 * true_positive + false_positive + false_negative
        class_scores.append(2 * true_positive / denominator if denominator else 0.0)

    return float(np.mean(class_scores))


def run(model_path: str, **model_kwargs: Any) -> None:
    train = pd.read_csv(TRAIN_PATH)
    combined = pd.read_csv(COMBINED_PATH)
    val = pd.read_csv(VALIDATION_PATH)
    test = pd.read_csv(TEST_PATH)

    X_train = _features(train, TRAIN_PATH, require_target=True)
    X_combined = _features(combined, COMBINED_PATH, require_target=True)
    X_validation = _features(val, VALIDATION_PATH, require_target=True)
    X_test = _features(test, TEST_PATH, require_target=False)
    if train[TARGET].isna().to_numpy().any():
        raise ValueError(f"Target column '{TARGET}' in '{TRAIN_PATH}' contains missing values.")
    if combined[TARGET].isna().to_numpy().any():
        raise ValueError(
            f"Target column '{TARGET}' in '{COMBINED_PATH}' contains missing values."
        )
    if val[TARGET].isna().to_numpy().any():
        raise ValueError(
            f"Target column '{TARGET}' in '{VALIDATION_PATH}' contains missing values."
        )

    expected_columns = X_train.columns.tolist()
    for source, features in (
        (COMBINED_PATH, X_combined),
        (VALIDATION_PATH, X_validation),
        (TEST_PATH, X_test),
    ):
        if features.columns.tolist() != expected_columns:
            raise ValueError(
                f"Feature columns in '{source}' must match training columns in the same order. "
                f"Train: {expected_columns}; found: {features.columns.tolist()}."
            )

    model = _load_model(model_path, **model_kwargs)
    _with_spinner(
        "Training model",
        lambda: model.fit(X_train.to_numpy(), train[TARGET].to_numpy()),
    )

    val_predictions = np.asarray(
        _with_spinner(
            "Predicting validation",
            lambda: model.predict(X_validation.to_numpy()),
        )
    )
    if val_predictions.shape != (len(val),):
        raise ValueError(
            "predict() must return one prediction per validation row; "
            f"got shape {val_predictions.shape}, expected {(len(val),)}."
        )
    macro_f1 = _macro_f1_score(
        val[TARGET].to_numpy(), val_predictions
    )

    final_model = _load_model(model_path, **model_kwargs)
    _with_spinner(
        "Retraining on combined",
        lambda: final_model.fit(
            X_combined.to_numpy(), combined[TARGET].to_numpy()
        ),
    )
    test_predictions = np.asarray(
        _with_spinner(
            "Predicting test",
            lambda: final_model.predict(X_test.to_numpy()),
        )
    )
    if test_predictions.shape != (len(test),):
        raise ValueError(
            "predict() must return one prediction per test row; "
            f"got shape {test_predictions.shape}, expected {(len(test),)}."
        )
    submission = pd.DataFrame(
        {"id": test["id"].to_numpy(), TARGET: test_predictions}
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"prediction_{timestamp}.csv"
    submission.to_csv(output_path, index=False)

    print(f"\nModel: {model_path}")
    if model_kwargs:
        params_str = ", ".join(f"{k}={v}" for k, v in model_kwargs.items())
        print(f"Parameters: {params_str}")
    print(f"Validation training rows: {len(train)}")
    print(f"Validation rows: {len(val)}")
    print(f"Final training rows: {len(combined)}")
    print(f"Validation Macro F1: {macro_f1:.4f}")
    print(f"Test predictions: {len(test_predictions)}")
    print(f"Saved submission (id, {TARGET}) to: {output_path}")


def _run_algorithm(name: str, **kwargs: Any) -> None:
    try:
        run(ALGORITHMS[name], **kwargs)
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
    parser = argparse.ArgumentParser(description="Evaluate SVM classifier and create a test submission.")
    parser.add_argument("-c", "--c", type=float, default=1.0, help="Regularization parameter C (default: 1.0)")
    parser.add_argument("--max-passes", type=int, default=3, help="Maximum iterations without change (default: 3)")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerance for stopping criterion (default: 1e-3)")
    parser.add_argument(
        "--kernel",
        type=str,
        default="linear",
        choices=["linear", "poly", "rbf", "gaussian"],
        help="Kernel type (default: linear)",
    )
    parser.add_argument("--degree", type=int, default=3, help="Degree of polynomial kernel (default: 3)")
    parser.add_argument(
        "--gamma",
        type=str,
        default="auto",
        help="Kernel coefficient for 'rbf', 'poly', and 'gaussian' ('auto', 'scale', or float value; default: auto)",
    )
    parser.add_argument("--coef0", type=float, default=0.0, help="Independent term in poly kernel (default: 0.0)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: None)")

    args = parser.parse_args()

    gamma_val: Any = args.gamma
    try:
        gamma_val = float(args.gamma)
    except ValueError:
        pass

    kwargs = {
        "c": args.c,
        "max_passes": args.max_passes,
        "tol": args.tol,
        "kernel": args.kernel,
        "degree": args.degree,
        "gamma": gamma_val,
        "coef0": args.coef0,
        "seed": args.seed,
    }
    _run_algorithm("svm", **kwargs)
