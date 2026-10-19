"""Download the benchmark datasets used in Table 2 of Liu et al. (2008).

Ten datasets come from ADBench (Han et al., NeurIPS 2022), which redistributes the
ODDS versions as .npz files with arrays X (features) and y (1 = anomaly).
Arrhythmia is not in ADBench, so it is built from the original UCI file using the
anomaly classes listed in the paper.

Mulcross (synthetic) is not downloaded; see README.md.
"""

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
ADBENCH_URL = "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/{}.npz"
UCI_ARRHYTHMIA_URL = "https://archive.ics.uci.edu/static/public/5/arrhythmia.zip"

# Name used in this project -> file name in ADBench
ADBENCH_FILES = {
    "Http": "16_http",
    "ForestCover": "10_cover",
    "Smtp": "34_smtp",
    "Shuttle": "32_shuttle",
    "Mammography": "23_mammography",
    "Annthyroid": "2_annthyroid",
    "Satellite": "30_satellite",
    "Pima": "29_Pima",
    "Breastw": "4_breastw",
    "Ionosphere": "18_Ionosphere",
}

# Arrhythmia classes treated as anomalies in Table 2 of the paper
ARRHYTHMIA_ANOMALY_CLASSES = {3, 4, 5, 7, 8, 9, 14, 15}


def download(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def get_adbench(name, file_name):
    target = DATA_DIR / f"{name}.npz"
    if target.exists():
        print(f"{name}: already downloaded")
        return
    data = np.load(io.BytesIO(download(ADBENCH_URL.format(file_name))))
    np.savez_compressed(target, X=data["X"].astype(float), y=data["y"].astype(int))
    print(f"{name}: saved")


def get_arrhythmia():
    target = DATA_DIR / "Arrhythmia.npz"
    if target.exists():
        print("Arrhythmia: already downloaded")
        return
    with zipfile.ZipFile(io.BytesIO(download(UCI_ARRHYTHMIA_URL))) as z:
        with z.open("arrhythmia.data") as f:
            df = pd.read_csv(f, header=None, na_values="?")
    labels = df.pop(df.columns[-1])
    # The UCI file has 279 attributes; dropping the 5 with missing values
    # leaves 274, which matches d = 274 in Table 2 of the paper.
    df = df.dropna(axis=1)
    X = df.to_numpy(dtype=float)
    y = labels.isin(ARRHYTHMIA_ANOMALY_CLASSES).astype(int).to_numpy()
    np.savez_compressed(target, X=X, y=y)
    print("Arrhythmia: saved")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    for name, file_name in ADBENCH_FILES.items():
        get_adbench(name, file_name)
    get_arrhythmia()


if __name__ == "__main__":
    main()
