## Highest kalian taro sini dong

| algo | f1 val | (kalo submit taro sini) | param |
| -- | -- | -- | --- |
| knn | 0.8210 | 0.83301 | `n_neighbors=22`, `weights="distance"` |
| dtl | 0.8350 | 0.8244 | `max_depth=15`, `min_samples_split=2`, `min_samples_leaf=2`, `class_weight={0: 1.0, 1: 1.2}`, `ccp_alpha=0.0001` |
| bayes | 0.7829 | 0.78483 | alpha=1.5, decision_threshold=0.475, n_bins=5, n_features=15, learn_structure=Tru |
| logistic regression | 0.81758 | 0.83382 | --learning-rate 0.02 --max-iter 380 --penalty elasticnet --reg-lambda 0.0007 --l1-ratio 0.18 --use-basis --n-landmarks 700 --rbf-gamma 0.18 --cont-rbf-gamma 0.30 --n-estimators 7 --avg-last-k 25 --threshold 0.46 --random-state 777 |
| svm | 0.8142 | 0.83186 | --kernel rbf -c 4.0 --gamma 0.08 --tol 0.015 --max-passes 2 --seed 42 |co