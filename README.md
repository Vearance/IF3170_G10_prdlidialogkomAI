# Tubes 2 AI - prdlidialogkomAI

Uses [uv](https://docs.astral.sh/uv/) to manage Python env and packages.

## commands

```bash
# Install or sync packages
uv sync
# Add a package
uv add <package-name>
```

## fast train-predict using [driver](src/driver.py)

Trains the selected algorithm with `data/clean/train.csv`, reports Macro F1 using `data/clean/val.csv`. Retrains `data/clean/combined.csv` and predicts `data/clean/test.csv`. Saves the predictions to a timestamped file in the root `output/` folder and a pickle model in `output/models/`.

```bash
uv run knn
uv run logres
uv run bayes
uv run dtl
uv run svm
```

## complete command examples for params

- [KNN](src/knn/README.md)
- [Logistic Regression](src/logres/README.md)
- [Naive Bayes](src/bayes/README.md)
- [Decision Tree](src/dtl/README.md#param-command-examples)
- [SVM](src/svm/README.md)


## predict using loaded model

```bash
uv run predict <model-name>.pkl
```

Only load `.pickle` files from `output/models/`.

## authors
| No. | Nama | NIM |
|---:|---|---|
| 1 | Irvin Tandiarrang Sumual | 13524030 |
| 2 | Bernhard Aprillio Pramana | 13524074 |
| 3 | Moreno Syawali Ganda Sugita | 13524096 |
| 4 | Jennifer Khang | 13524110 |
| 5 | Nathaniel Christian | 13524122 |