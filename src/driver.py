"""Evaluate a classifier on validation data and create a test submission."""

from __future__ import annotations

import argparse
import ast
import importlib
import itertools
import pickle
import re
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
MODEL_DIR = OUTPUT_DIR / "models"
VIZ_DIR = OUTPUT_DIR / "viz"
TARGET = "is_canceled"
IGNORED_COLUMNS = ["id"]
ALGORITHMS = {
    "knn": "src.knn.alg:KNN",
    "logres": "src.logres.alg:LogisticRegression",
    "bayes": "src.bayes.alg:NaiveBayes",
    "dtl": "src.dtl.alg:DecisionTree",
    "svm": "src.svm.alg:SVM",
    "svm_kelas": "src.svm.svm_kelas:SVMKelas",
}


def _load_model(model_path: str, **kwargs: Any) -> Any:
    """Load and instantiate a model class specified as ``module.path:ClassName``."""
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



def _parse_optional_int(raw: str) -> int | None:
    if raw.strip().lower() in {"none", "null"}:
        return None
    try:
        return int(raw)
    except ValueError as err:
        raise argparse.ArgumentTypeError("Enter an integer or 'none'.") from err


def _parse_class_weight(raw: str) -> dict[Any, float] | str | None:
    text = raw.strip()
    if text.lower() in {"none", "null"}:
        return None
    if text.lower() == "balanced":
        return "balanced"
    try:
        weights = ast.literal_eval(text)
    except (ValueError, SyntaxError) as err:
        raise argparse.ArgumentTypeError(
            "Enter 'balanced', 'none', or a dictionary such as '{0: 1.0, 1: 1.2}'."
        ) from err
    if not isinstance(weights, dict):
        raise argparse.ArgumentTypeError(
            "class-weight must be 'balanced', 'none', or a dictionary."
        )
    try:
        return {label: float(weight) for label, weight in weights.items()}
    except (TypeError, ValueError) as err:
        raise argparse.ArgumentTypeError("Dictionary weights must be numeric.") from err


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


def _submission_filename(
    model_path: str, model_kwargs: dict[str, Any], timestamp: str
) -> str:
    """Build a readable, filesystem-safe submission filename from model parameters."""
    module_path, _, class_name = model_path.partition(":")
    module_parts = module_path.split(".")
    model_name = module_parts[-2] if len(module_parts) > 1 else class_name.lower()

    def slug(value: Any) -> str:
        text = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value)).strip("-._")
        return text or "empty"

    parameter_parts = [
        f"{slug(name)}-{slug(value)}"
        for name, value in sorted(model_kwargs.items())
    ]
    suffix = f"_{'_'.join(parameter_parts)}" if parameter_parts else ""
    return f"{model_name}{suffix}_{timestamp}.csv"


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
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / _submission_filename(model_path, model_kwargs, timestamp)
    model_output_path = MODEL_DIR / output_path.with_suffix(".pkl").name
    with model_output_path.open("wb") as model_file:
        pickle.dump(final_model, model_file, protocol=pickle.HIGHEST_PROTOCOL)
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
    print(f"Saved model pickle to: {model_output_path}")
    print(f"Saved submission (id, {TARGET}) to: {output_path}")


def run_saved_model(model_filename: str) -> None:
    """Load a trusted pickle from output/models and predict the test data."""
    filename = Path(model_filename)
    if filename.name != model_filename or filename.suffix.lower() != ".pkl":
        raise ValueError("Provide only a .pkl filename located in output/models.")

    models_root = MODEL_DIR.resolve()
    model_path = (MODEL_DIR / filename).resolve()
    if models_root not in model_path.parents:
        raise ValueError("The model file must be inside output/models.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model was not found: {model_path}")

    with model_path.open("rb") as model_file:
        model = _with_spinner("Loading saved model", lambda: pickle.load(model_file))
    if not callable(getattr(model, "predict", None)):
        raise TypeError(f"Saved object in '{model_path}' does not provide predict().")

    test = pd.read_csv(TEST_PATH)
    X_test = _features(test, TEST_PATH, require_target=False)
    predictions = np.asarray(
        _with_spinner(
            "Predicting test",
            lambda: model.predict(X_test.to_numpy()),
        )
    )
    if predictions.shape != (len(test),):
        raise ValueError(
            "predict() must return one prediction per test row; "
            f"got shape {predictions.shape}, expected {(len(test),)}."
        )

    submission = pd.DataFrame({"id": test["id"].to_numpy(), TARGET: predictions})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"{model_path.stem}_prediction_{timestamp}.csv"
    submission.to_csv(output_path, index=False)
    print(f"Loaded model: {model_path}")
    print(f"Test predictions: {len(predictions)}")
    print(f"Saved submission (id, {TARGET}) to: {output_path}")


def _run_algorithm(name: str, **kwargs: Any) -> None:
    try:
        run(ALGORITHMS[name], **kwargs)
    except (OSError, ValueError, TypeError) as err:
        raise SystemExit(f"Error running {name}: {err}") from err


def knn_main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate K-Nearest Neighbors and create a test submission."
    )
    parser.add_argument("--n-neighbors", type=int, default=7)
    parser.add_argument(
        "--weights", choices=["uniform", "distance"], default="distance"
    )
    parser.add_argument(
        "--metric",
        choices=["euclidean", "correlation"],
        default="euclidean",
        help="Distance metric (default: euclidean).",
    )
    args = parser.parse_args()
    _run_algorithm(
        "knn",
        n_neighbors=args.n_neighbors,
        weights=args.weights,
        metric=args.metric,
    )


def logres_main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Logistic Regression and create a test submission."
    )
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument(
        "--penalty", choices=["none", "l1", "l2", "elasticnet"], default="none"
    )
    parser.add_argument("--reg-lambda", type=float, default=0.0001)
    parser.add_argument("--l1-ratio", type=float, default=0.5)
    parser.add_argument("--use-basis", action="store_true")
    parser.add_argument("--continuous-cols", type=int, default=14)
    parser.add_argument("--pca-cols", type=int, default=12)
    parser.add_argument("--n-landmarks", type=int, default=0)
    parser.add_argument("--rbf-gamma", type=float, default=0.15)
    parser.add_argument("--cont-rbf-gamma", type=float, default=0.25)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--avg-last-k", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--neg-weight", type=float, default=1.0)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate a log-loss contour and training trajectory video.",
    )
    args = parser.parse_args()

    kwargs = vars(args)
    visualize = kwargs.pop("visualize")
    if kwargs["penalty"] == "none":
        kwargs["penalty"] = None
    _run_algorithm("logres", **kwargs)

    if visualize:
        from src.logres.visualize import generate_contour_video

        video_path = _with_spinner(
            "Rendering Logistic Regression contour video",
            lambda: generate_contour_video(
                learning_rate=args.learning_rate,
                max_iter=args.max_iter,
                output_dir=VIZ_DIR,
            ),
        )
        print(f"Saved Logistic Regression visualization to: {video_path}")


def bayes_main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Naive Bayes and create a test submission."
    )
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--threshold", type=float, default=0.475)
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--n-features", type=_parse_optional_int, default=15)
    parser.add_argument(
        "--learn-structure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Learn TAN-style feature dependencies (use --no-learn-structure for Naive Bayes)",
    )
    parser.add_argument(
        "--visualize",
        "--kak_aku_mw_gambarnya_boleh_gk",
        action="store_true",
        help="Generate a graph of relationships between class and features",
    )
    args = parser.parse_args()

    kwargs = {
        "alpha": args.alpha,
        "decision_threshold": args.threshold,
        "n_bins": args.n_bins,
        "n_features": args.n_features,
        "learn_structure": args.learn_structure,
    }
    _run_algorithm("bayes", **kwargs)

    if args.visualize:
        from src.bayes.visualize import generate_bayes_structure_plot

        vis_path = _with_spinner(
            "Rendering Bayesian network plot",
            lambda: generate_bayes_structure_plot(**kwargs, output_dir=VIZ_DIR),
        )
        print(f"Saved Bayesian network visualization to: {vis_path}")


def dtl_main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Decision Tree and create a test submission."
    )
    parser.add_argument("--max-depth", type=_parse_optional_int, default=15)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument(
        "--class-weight",
        type=_parse_class_weight,
        default={0: 1.0, 1: 1.2},
        help="Class weights as a dictionary, 'balanced', or 'none'.",
    )
    parser.add_argument("--ccp-alpha", type=float, default=0.00010)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Print and save an ASCII visualization of the fitted decision tree.",
    )
    args = parser.parse_args()
    model_kwargs = vars(args)
    visualize = model_kwargs.pop("visualize")
    _run_algorithm("dtl", **model_kwargs)

    if visualize:
        from src.dtl.visualize import print_tree

        train = pd.read_csv(TRAIN_PATH)
        X_train = _features(train, TRAIN_PATH, require_target=True)
        visualization_model = _load_model(ALGORITHMS["dtl"], **model_kwargs)
        _with_spinner(
            "Training visualization tree",
            lambda: visualization_model.fit(
                X_train.to_numpy(), train[TARGET].to_numpy()
            ),
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        visualization_path = VIZ_DIR / f"dtl_tree_{timestamp}.txt"
        print_tree(
            visualization_model,
            feature_names=X_train.columns.tolist(),
            save_path=visualization_path,
        )


def svm_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SVM classifier and create a test submission.")
    parser.add_argument(
        "--impl",
        type=str,
        default="keerthi",
        choices=["keerthi", "kelas"],
        help="SVM implementation: 'keerthi' (default) or 'kelas' (QP solver from PPT)",
    )
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
    parser.add_argument("--max-samples", type=int, default=None, help="Max training samples to subsample (default: None for full data)")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate 2D decision boundary, margin, and support vector plots (Linear, Poly, Gaussian)",
    )

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
        "max_samples": args.max_samples,
    }
    algo_key = "svm_kelas" if args.impl == "kelas" else "svm"
    _run_algorithm(algo_key, **kwargs)

    if args.visualize:
        from src.svm.visualize import generate_svm_kernel_plots

        vis_seed = 42 if args.seed is None else args.seed
        vis_path = _with_spinner(
            "Rendering SVM kernel plot",
            lambda: generate_svm_kernel_plots(
                seed=vis_seed,
                kernel=args.kernel,
                c=args.c,
                degree=args.degree,
                gamma=gamma_val,
                coef0=args.coef0,
                max_passes=args.max_passes,
                tol=args.tol,
                output_dir=VIZ_DIR,
            ),
        )
        print(f"Saved SVM visualization to: {vis_path}")


def run_saved_model_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a saved model pickle from output/models on the test data."
    )
    parser.add_argument(
        "model_filename",
        help="Pickle filename inside output/models (for example: knn_20260101_120000.pkl).",
    )
    args = parser.parse_args()
    try:
        run_saved_model(args.model_filename)
    except (OSError, ValueError, TypeError, pickle.PickleError, EOFError) as err:
        raise SystemExit(f"Error running saved model: {err}") from err


def svm_kelas_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SVM Kelas (Slide PPT QP Solver) classifier and create a test submission.")
    parser.add_argument("-c", "--c", type=float, default=1.0, help="Regularization parameter C (default: 1.0)")
    parser.add_argument("--tol", type=float, default=1e-4, help="Tolerance for QP stopping criterion (default: 1e-4)")
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        choices=["linear", "poly", "rbf", "gaussian"],
        help="Kernel type (default: rbf)",
    )
    parser.add_argument("--degree", type=int, default=3, help="Degree of polynomial kernel (default: 3)")
    parser.add_argument(
        "--gamma",
        type=str,
        default="scale",
        help="Kernel coefficient ('auto', 'scale', or float value; default: scale)",
    )
    parser.add_argument("--coef0", type=float, default=0.0, help="Independent term in poly kernel (default: 0.0)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: None)")
    parser.add_argument("--max-samples", type=int, default=1000, help="Max training samples for QP optimization (default: 1000)")

    args = parser.parse_args()

    gamma_val: Any = args.gamma
    try:
        gamma_val = float(args.gamma)
    except ValueError:
        pass

    kwargs = {
        "c": args.c,
        "tol": args.tol,
        "kernel": args.kernel,
        "degree": args.degree,
        "gamma": gamma_val,
        "coef0": args.coef0,
        "seed": args.seed,
        "max_samples": args.max_samples,
    }
    _run_algorithm("svm_kelas", **kwargs)

