"""Evaluate a classifier on validation data and create a test submission."""

from __future__ import annotations

import ast
import importlib
import inspect
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


def _load_model_class(model_path: str) -> type[Any]:
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
    return model_class


def _load_model(model_path: str, **kwargs: Any) -> Any:
    """Instantiate a model class specified as ``module.path:ClassName`` with kwargs."""
    model_class = _load_model_class(model_path)
    return model_class(**kwargs)


def _parse_value(raw: str) -> Any:
    """Convert a string CLI/input value into int, float, bool, None, tuple, or str."""
    text = raw.strip()
    lower = text.lower()
    if lower in {"none", "null"}:
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _get_model_parameters(model_path: str) -> dict[str, inspect.Parameter]:
    """Inspect __init__ parameters of the target model class."""
    model_class = _load_model_class(model_path)
    sig = inspect.signature(model_class.__init__)
    return {
        name: param
        for name, param in sig.parameters.items()
        if name != "self"
        and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }


def _resolve_model_kwargs(model_path: str, argv: list[str] | None = None) -> dict[str, Any]:
    """Parse CLI arguments or interactively prompt for model parameters in the terminal."""
    args = list(sys.argv[1:] if argv is None else argv)
    params = _get_model_parameters(model_path)
    _, class_name = model_path.split(":", maxsplit=1)

    if any(arg in {"-h", "--help"} for arg in args):
        print(f"Model: {model_path}")
        print("\nAvailable parameters:")
        for name, param in params.items():
            default = param.default if param.default is not inspect.Parameter.empty else "<required>"
            flag = f"--{name.replace('_', '-')}"
            print(f"  {flag:<22} (or {name}=<val>)  default: {default!r}")
        print("\nFlags:")
        print("  -y, --default          Run immediately with default parameters (no prompt)")
        print("  -i, --interactive      Prompt interactively for each parameter")
        if model_path == ALGORITHMS["logres"]:
            print("  --visualize            Generate log-loss contour & trajectory .avi video")
        raise SystemExit(0)

    use_defaults_flag = False
    force_interactive = False
    filtered_args: list[str] = []
    for arg in args:
        if arg in {"-y", "--yes", "--default"}:
            use_defaults_flag = True
        elif arg in {"-i", "--interactive"}:
            force_interactive = True
        else:
            filtered_args.append(arg)

    kwargs: dict[str, Any] = {}
    i = 0
    while i < len(filtered_args):
        token = filtered_args[i]
        if token.startswith("--"):
            body = token[2:]
            if "=" in body:
                key_raw, val_raw = body.split("=", maxsplit=1)
                i += 1
            else:
                key_raw = body
                if i + 1 >= len(filtered_args):
                    raise ValueError(f"Option '{token}' requires a value.")
                val_raw = filtered_args[i + 1]
                i += 2
            key = key_raw.replace("-", "_")
        elif "=" in token:
            key_raw, val_raw = token.split("=", maxsplit=1)
            key = key_raw.replace("-", "_")
            i += 1
        else:
            raise ValueError(
                f"Unrecognized argument '{token}'. Use '--param value' or 'param=value'."
            )

        if key not in params:
            valid_list = ", ".join(params.keys())
            raise ValueError(
                f"Unknown parameter '{key}' for {class_name}. Valid parameters: {valid_list}."
            )
        kwargs[key] = _parse_value(val_raw)

    # Interactive prompt if running in an interactive terminal without CLI overrides
    if (force_interactive or (not kwargs and not use_defaults_flag and sys.stdin.isatty())) and params:
        print(f"\n=== Parameter Setup: {class_name} ===")
        print("Masukkan nilai parameter (tekan Enter untuk menggunakan nilai default):")
        for name, param in params.items():
            if name in kwargs:
                continue
            default = param.default if param.default is not inspect.Parameter.empty else None
            try:
                user_input = input(f"  {name} [{default!r}]: ").strip()
            except EOFError:
                break
            if user_input != "":
                kwargs[name] = _parse_value(user_input)
        print()

    return kwargs


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


def run(model_path: str, model_kwargs: dict[str, Any] | None = None) -> None:
    kwargs = model_kwargs or {}
    # Validate model instantiation upfront before loading CSVs
    model = _load_model(model_path, **kwargs)

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

    final_model = _load_model(model_path, **kwargs)
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

    active_params = {
        name: getattr(final_model, name, param.default)
        for name, param in _get_model_parameters(model_path).items()
    }
    print(f"\nModel: {model_path}")
    print(f"Parameters: {active_params}")
    print(f"Validation training rows: {len(train)}")
    print(f"Validation rows: {len(val)}")
    print(f"Final training rows: {len(combined)}")
    print(f"Validation Macro F1: {macro_f1:.4f}")
    print(f"Test predictions: {len(test_predictions)}")
    print(f"Saved submission (id, {TARGET}) to: {output_path}")


def _run_algorithm(name: str, argv: list[str] | None = None) -> None:
    try:
        raw_args = list(sys.argv[1:] if argv is None else argv)
        visualize = False
        clean_args: list[str] = []
        for arg in raw_args:
            if arg in {"--visualize", "--video"}:
                visualize = True
            else:
                clean_args.append(arg)

        model_path = ALGORITHMS[name]
        model_kwargs = _resolve_model_kwargs(model_path, argv=clean_args)
        run(model_path, model_kwargs=model_kwargs)

        if visualize:
            if name != "logres":
                raise ValueError("--visualize is currently only supported for 'logres'.")
            from src.logres.visualize import generate_contour_video

            vid_kwargs: dict[str, Any] = {}
            if "learning_rate" in model_kwargs:
                vid_kwargs["learning_rate"] = float(model_kwargs["learning_rate"])
            if "max_iter" in model_kwargs:
                vid_kwargs["max_iter"] = int(model_kwargs["max_iter"])
            video_path = _with_spinner(
                "Rendering contour video",
                lambda: generate_contour_video(**vid_kwargs),
            )
            print(f"Saved contour video to: {video_path}")
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


def main() -> None:
    """CLI entrypoint supporting `python -m src.driver <algorithm> [params...]`."""
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help"} and len(args) == 1:
        available = ", ".join(ALGORITHMS.keys())
        print(f"Usage: python -m src.driver <algorithm> [--param value | param=value ...]")
        print(f"Available algorithms: {available}")
        raise SystemExit(0)

    alg_name = args[0]
    if alg_name not in ALGORITHMS:
        available = ", ".join(ALGORITHMS.keys())
        raise SystemExit(f"Unknown algorithm '{alg_name}'. Choose from: {available}")
    _run_algorithm(alg_name, argv=args[1:])


if __name__ == "__main__":
    main()
