# BRAF
Implement random forest and BRAF from scratch


To train a random forest (RF) or Biased Random Forest (BRAF) on the diabetes dataset with k-fold cross validation, and to save ROC and PRC curves use:

```python evaluate_forest.py --forest BRAF --k 10 --folds 10 --p_ratio 0.5 --size 100```

Arguments:

```--forest:   BRAF or RF```

```--k:        BRAF argument k, for k-nearest neighbors in cirical data set```

```--folds:    Number of folds for cross-validation```

```--p_ratio:  BRAF argument. Relative ratio of random forests trained on critical/full data set```

```--size 100: Number of trees in random forest```


References:
https://ieeexplore.ieee.org/document/8541100
