# Reproduction of Isolation Forest (Liu, Ting & Zhou, ICDM 2008)

Reproduction of the main experiment (Table 3, iForest and LOF columns) of:

> F. T. Liu, K. M. Ting, and Z.-H. Zhou. *Isolation Forest.* In Proceedings of the 8th IEEE
> International Conference on Data Mining (ICDM), pp. 413–422, 2008. doi:10.1109/ICDM.2008.17

Assignment 2, Data Wrangling (INFOMDW), Utrecht University.

## Files

| File | Purpose |
|---|---|
| `download_data.py` | Downloads the datasets into `data/` |
| `run_experiment.py` | Runs iForest (and optionally LOF) and writes `results/results.csv` and `results/results.md` |
| `requirements.txt` | Python package versions used |
| `results/` | Results of the run reported in the report |

## How to run

Tested with Python 3.14 on Windows 11.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
python download_data.py
python run_experiment.py --with-lof
```

The full run (11 datasets × 10 repetitions + LOF) takes about 2 minutes on a laptop.

Other options:

```bash
python run_experiment.py --runs 3 --datasets Pima Breastw   # quick test
python run_experiment.py --trees 100 --psi 256               # change parameters
python run_experiment.py --help
```

## Parameters

| Parameter | Value | Same as paper? |
|---|---|---|
| Implementation | `sklearn.ensemble.IsolationForest` (scikit-learn 1.9.1) | Paper used the authors' own implementation |
| Number of trees `t` (`n_estimators`) | 100 | Yes |
| Sub-sampling size `ψ` (`max_samples`) | 256 (or `n` if the dataset is smaller) | Yes |
| Tree height limit | ceiling(log2 ψ) = 8 (set automatically by scikit-learn) | Yes |
| Attributes per tree (`max_features`) | 1.0 (all attributes) | Yes |
| Sampling (`bootstrap`) | False (without replacement) | Yes |
| Threads (`n_jobs`) | 1 | Yes (single-threaded) |
| `contamination` | default; it does not affect the scores or the AUC | n/a |
| Training / evaluation data | all instances; labels used only for the AUC | Yes |
| Repetitions | 10, seeds 0–9; mean ± standard deviation reported | Not stated in the paper |
| LOF `k` (`n_neighbors`) | 10 | Yes |

## Data

Ten datasets come from [ADBench](https://github.com/Minqi824/ADBench) (`adbench/datasets/Classical`),
which redistributes the ODDS versions. Arrhythmia is built from the
[UCI file](https://archive.ics.uci.edu/dataset/5/arrhythmia): classes 3, 4, 5, 7, 8, 9, 14, 15 are
anomalies (as in the paper) and the 5 attributes with missing values are dropped, giving 274 attributes.

| Dataset | n (ours) | n (paper) | d (ours) | d (paper) | Anomalies (ours) | Anomalies (paper) |
|---|---|---|---|---|---|---|
| Http | 567,498 | 567,497 | 3 | 3 | 0.39% | 0.4% |
| ForestCover | 286,048 | 286,048 | 10 | 10 | 0.96% | 0.9% |
| Smtp | 95,156 | 95,156 | 3 | 3 | 0.03% | 0.03% |
| Shuttle | 49,097 | 49,097 | 9 | 9 | 7.15% | 7% |
| Mammography | 11,183 | 11,183 | 6 | 6 | 2.32% | 2% |
| Annthyroid | 7,200 | 6,832 | 6 | 6 | 7.42% | 7% |
| Satellite | 6,435 | 6,435 | 36 | 36 | 31.64% | 32% |
| Pima | 768 | 768 | 8 | 8 | 34.90% | 35% |
| Breastw | 683 | 683 | 9 | 9 | 34.99% | 35% |
| Arrhythmia | 452 | 452 | 274 | 274 | 14.60% | 15% |
| Ionosphere | 351 | 351 | 32 | 32 | 35.90% | 36% |

Not included: **Mulcross** (synthetic data from a generator that is not publicly available in its original
form). ORCA and Random Forests from Table 3 are also not reproduced.

## Reused code, data and licenses

No external code was copied into this repository. The two scripts were written for this project
(with the help of an AI assistant, Claude; see the report's AI disclosure). They reuse the following
software and data, all under licenses that allow use, modification and redistribution:

| What | How it is used | License |
|---|---|---|
| [scikit-learn](https://scikit-learn.org) (`IsolationForest`, `LocalOutlierFactor`, `roc_auc_score`) | Imported as a library, unmodified | BSD-3-Clause |
| [ADBench](https://github.com/Minqi824/ADBench) datasets (ODDS versions) | Downloaded at run time by `download_data.py`; not included in this repository | BSD-2-Clause |
| [UCI Arrhythmia](https://archive.ics.uci.edu/dataset/5/arrhythmia) dataset | Downloaded at run time by `download_data.py`; not included in this repository | CC BY 4.0 |

The code in this repository is released under the MIT License (see `LICENSE`).
