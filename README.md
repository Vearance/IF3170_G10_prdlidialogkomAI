# Tubes 2 AI

Uses [uv](https://docs.astral.sh/uv/) to manage Python env and packages.

## commands

```bash
# Install or sync packages
uv sync
# Add a package
uv add <package-name>
```

## fast train-predict using [driver](src/driver.py)

Trains the selected algorithm with `data/clean/train.csv` and predicts labels for `data/clean/test.csv`. Saves the predictions to a timestamped file in the root `output/` folder.

```bash
uv run knn
uv run logres
uv run bayes
uv run dtl
uv run svm
```

## authors
| No. | Nama | NIM |
|---:|---|---|
| 1 | - | 13524030 |
| 2 | Bernhard Aprillio Pramana | 13524074 |
| 3 | - | 13524096 |
| 4 | Jennifer Khang | 13524110 |
| 5 | Nathaniel Christian | 13524122 |