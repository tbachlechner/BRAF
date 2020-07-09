# BRAF
Implement random forest and BRAF from scratch


To train a random forest (RF) or Biased Random Forest (BRAF) on the diabetes dataset with k-fold cross validation, and to save ROC and PRC curves use:

```python evaluate_forest.py --forest BRAF --k 10 --folds 10 --p_ratio 0.5 --size 100```

Arguments:

```
--forest:   BRAF or RF
--k:        BRAF argument k, for k-nearest neighbors in cirical data set
--folds:    Number of folds for cross-validation
--p_ratio:  BRAF argument. Relative ratio of random forests trained on critical/full data set
--size 100: Number of trees in random forest
```

Sample outputs:

BRAF evaluation
```
>>>  python evaluate_forest.py --forest BRAF --k 10 --folds 10 --p_ratio 0.5 --size 100
------------------------------------------------------------------------------------------
| Fold:  0 | Accuracy: 0.74 | Precision: 0.60 | Recall: 0.60 | AUROC: 0.74 | AUPRC: 0.54 |
| Fold:  1 | Accuracy: 0.73 | Precision: 0.63 | Recall: 0.61 | AUROC: 0.80 | AUPRC: 0.60 |
| Fold:  2 | Accuracy: 0.66 | Precision: 0.44 | Recall: 0.52 | AUROC: 0.73 | AUPRC: 0.55 |
| Fold:  3 | Accuracy: 0.69 | Precision: 0.53 | Recall: 0.62 | AUROC: 0.77 | AUPRC: 0.55 |
| Fold:  4 | Accuracy: 0.83 | Precision: 0.83 | Recall: 0.75 | AUROC: 0.88 | AUPRC: 0.83 |
| Fold:  5 | Accuracy: 0.71 | Precision: 0.62 | Recall: 0.67 | AUROC: 0.75 | AUPRC: 0.68 |
| Fold:  6 | Accuracy: 0.77 | Precision: 0.70 | Recall: 0.59 | AUROC: 0.84 | AUPRC: 0.75 |
| Fold:  7 | Accuracy: 0.73 | Precision: 0.65 | Recall: 0.42 | AUROC: 0.85 | AUPRC: 0.64 |
| Fold:  8 | Accuracy: 0.78 | Precision: 0.71 | Recall: 0.63 | AUROC: 0.83 | AUPRC: 0.74 |
| Fold:  9 | Accuracy: 0.84 | Precision: 0.83 | Recall: 0.62 | AUROC: 0.85 | AUPRC: 0.75 |
------------------------------------------------------------------------------------------

Averages +/- Standard Error
---------------------------
 Accuracy:  0.75 +/- 0.02
 Precision: 0.65 +/- 0.04
 Recall:    0.60 +/- 0.03
 AUROC:     0.80 +/- 0.02
 AUPRC:     0.66 +/- 0.03

```
<p align="left">
<img src=./data/BRAF_prc_fold9.png width="350"/><img src=./data/BRAF_roc_fold9.png width="350"/>
</p>


RF evaluation
```
>>>  python evaluate_forest.py --forest RF --size 100
------------------------------------------------------------------------------------------
| Fold:  0 | Accuracy: 0.69 | Precision: 0.54 | Recall: 0.50 | AUROC: 0.66 | AUPRC: 0.60 |
| Fold:  1 | Accuracy: 0.75 | Precision: 0.57 | Recall: 0.59 | AUROC: 0.75 | AUPRC: 0.60 |
| Fold:  2 | Accuracy: 0.79 | Precision: 0.77 | Recall: 0.72 | AUROC: 0.73 | AUPRC: 0.78 |
| Fold:  3 | Accuracy: 0.75 | Precision: 0.65 | Recall: 0.58 | AUROC: 0.72 | AUPRC: 0.65 |
| Fold:  4 | Accuracy: 0.73 | Precision: 0.61 | Recall: 0.63 | AUROC: 0.66 | AUPRC: 0.60 |
| Fold:  5 | Accuracy: 0.77 | Precision: 0.82 | Recall: 0.48 | AUROC: 0.70 | AUPRC: 0.72 |
| Fold:  6 | Accuracy: 0.77 | Precision: 0.58 | Recall: 0.68 | AUROC: 0.69 | AUPRC: 0.60 |
| Fold:  7 | Accuracy: 0.73 | Precision: 0.62 | Recall: 0.56 | AUROC: 0.65 | AUPRC: 0.62 |
| Fold:  8 | Accuracy: 0.79 | Precision: 0.69 | Recall: 0.74 | AUROC: 0.77 | AUPRC: 0.79 |
| Fold:  9 | Accuracy: 0.73 | Precision: 0.73 | Recall: 0.53 | AUROC: 0.66 | AUPRC: 0.77 |
------------------------------------------------------------------------------------------

Averages +/- Standard Error
---------------------------
 Accuracy:  0.75 +/- 0.01
 Precision: 0.66 +/- 0.03
 Recall:    0.60 +/- 0.03
 AUROC:     0.70 +/- 0.01
 AUPRC:     0.67 +/- 0.02
 ```
<p align="left">
<img src=./data/RF_prc_fold9.png width="350"/><img src=./data/RF_roc_fold9.png width="350"/>
</p>


References:
https://ieeexplore.ieee.org/document/8541100
