## distance metrics

KNN supports `euclidean` (default) and `correlation`, which weights standardized features by their correlation with training labels.

## param command example

```bash
uv run knn --n-neighbors 22 --weights distance --metric correlation
```
