## distance metrics

KNN supports two distance metrics:

- `euclidean`: the original Euclidean distance, with equal feature weights.
- `correlation`: standardizes each feature and weights it by its absolute correlation with the training labels, reducing the influence of weakly predictive features.

Correlation weights are estimated using training data only. The default remains
Euclidean to preserve existing behavior.

## param command example

```bash
uv run knn --n-neighbors 22 --weights distance --metric correlation
```
