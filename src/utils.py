"""Utilidades compartidas para los notebooks del repo.

Uso:
    import sys; sys.path.append("../..")
    from src.utils import set_seed, download_if_missing, plot_confusion_matrix
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

RANDOM_STATE = 42


def set_seed(seed: int = RANDOM_STATE) -> int:
    """Fija semillas de random/numpy para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    return seed


def download_if_missing(url: str, dest: str | Path, timeout: int = 120) -> Path:
    """Descarga un archivo solo si no existe localmente (cache en data/raw).

    Evita usar `!wget` dentro de los notebooks. Los CSV grandes estan
    ignorados por .gitignore; solo se versiona una muestra pequena.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[cache] {dest} ya existe, no se descarga.")
        return dest
    try:
        import urllib.request

        print(f"[download] {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
    except Exception as e:  # noqa: BLE001 - queremos mensaje claro en notebook
        print(f"[warn] No se pudo descargar {url}: {e}")
        print("       Revisa tu conexion o descarga manual a", dest)
    return dest


def load_csv_cached(url: str, local_name: str, **read_csv_kwargs) -> pd.DataFrame:
    """Lee un CSV remoto con cache local en data/raw/.

    Ejemplo:
        df = load_csv_cached(URL_CHURN, "ChurnData.csv")
    """
    # Resuelve data/raw tanto si el notebook esta en raiz como en notebooks/*/
    here = Path.cwd()
    candidates = [here / "data" / "raw", here.parent / "data" / "raw", here.parent.parent / "data" / "raw"]
    raw_dir = next((p for p in candidates if p.exists() or p.parent.exists()), here / "data" / "raw")
    dest = raw_dir / local_name
    download_if_missing(url, dest)
    if dest.exists():
        return pd.read_csv(dest, **read_csv_kwargs)
    # Fallback: lectura directa (requiere internet)
    print("[fallback] Leyendo directamente desde URL")
    return pd.read_csv(url, **read_csv_kwargs)


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Matriz de confusion"):
    """Dibuja matriz de confusion con sklearn (reemplaza funciones ad-hoc)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    _, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return cm
