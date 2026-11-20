"""Utilidades compartidas para los notebooks del repo.

Uso:
    import sys; sys.path.append("../..")
    from src.utils import set_seed, download_if_missing, load_csv_cached, plot_confusion_matrix
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)
from sklearn.decomposition import PCA

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
    candidates = [
        here / "data" / "raw",
        here.parent / "data" / "raw",
        here.parent.parent / "data" / "raw",
    ]
    # Priorizar el directorio que YA existe
    raw_dir = next((p for p in candidates if p.is_dir()), None)
    if raw_dir is None:
        # Crear en la ubicación más probable (data/raw relativo a la raíz del repo)
        raw_dir = here.parent.parent / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / local_name
    # Si ya existe en caché local, leer directamente (sin intentar descargar)
    if dest.exists():
        print(f"[cache] {dest} ya existe, se lee desde caché local.")
        return pd.read_csv(dest, **read_csv_kwargs)
    # Si no existe localmente, intentar descargar (requiere internet)
    download_if_missing(url, dest)
    if dest.exists():
        return pd.read_csv(dest, **read_csv_kwargs)
    # Fallback: lectura directa (requiere internet)
    print("[fallback] Leyendo directamente desde URL")
    return pd.read_csv(url, **read_csv_kwargs)


# ---------------------------------------------------------------------------
# Visualizaciones avanzadas y reutilizables
# ---------------------------------------------------------------------------

def plot_regression_future(
    x_train,
    y_train,
    model,
    x_future,
    x_label: str = "X",
    y_label: str = "y",
    title: str = "Regresión — predicción de valores futuros",
    ci: bool = False,
):
    """Dibuja puntos de entrenamiento, línea de ajuste y predicción futura.

    Parameters
    ----------
    x_train, y_train : array-like
        Datos de entrenamiento (1D).
    model : estimador sklearn ajustado con metodo ``predict``.
    x_future : array-like
        Valores futuros de X para predecir (extrapolación/interpolación).
    ci : bool
        Si True, muestra banda de intervalo de confianza aproximada
        (desviación estándar de los residuos de train).
    """
    x_train = np.asarray(x_train).ravel()
    y_train = np.asarray(y_train).ravel()
    x_future = np.asarray(x_future).ravel()

    y_pred_future = np.asarray(model.predict(x_future.reshape(-1, 1))).ravel()

    plt.figure(figsize=(9, 5))
    plt.scatter(x_train, y_train, alpha=0.5, label="Entrenamiento (real)")
    x_plot = np.linspace(x_future.min(), x_future.max(), 200)
    plt.plot(
        x_plot,
        np.asarray(model.predict(x_plot.reshape(-1, 1))).ravel(),
        color="red",
        linewidth=2,
        label="Línea de ajuste",
    )
    plt.scatter(
        x_future,
        y_pred_future,
        color="green",
        marker="X",
        s=70,
        label="Predicción futura",
        zorder=5,
    )
    if ci:
        resid = y_train - model.predict(x_train.reshape(-1, 1))
        std = resid.std()
        plt.fill_between(
            x_plot,
            model.predict(x_plot.reshape(-1, 1)) - std,
            model.predict(x_plot.reshape(-1, 1)) + std,
            color="red",
            alpha=0.15,
            label="±1 std (residuos train)",
        )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return y_pred_future


def plot_residuals(y_true, y_pred, title: str = "Residuos vs valores predichos"):
    """Grafica residuos vs valores predichos para detectar sesgos/heteroscedasticidad."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    resid = y_true - y_pred

    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, resid, alpha=0.6, edgecolor="k", linewidth=0.4)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Valor predicho")
    plt.ylabel("Residuo (real − predicho)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return resid


def plot_roc_curve(y_true, y_score, pos_label=1, title: str = "Curva ROC"):
    """Dibuja la curva ROC y devuelve el AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Azar")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return roc_auc


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


def plot_elbow_silhouette(X, k_range=range(2, 11), title_prefix: str = "K-Means"):
    """Curva del codo (inercia) y coeficiente de silueta para elegir k."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    inertias = []
    silhouettes = []
    ks = list(k_range)

    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(ks, inertias, marker="o", color="steelblue")
    axes[0].set_xlabel("k (n.º de clusters)")
    axes[0].set_ylabel("Inercia")
    axes[0].set_title(f"{title_prefix} — Método del codo")
    axes[0].grid(alpha=0.3)

    axes[1].plot(ks, silhouettes, marker="o", color="seagreen")
    axes[1].set_xlabel("k (n.º de clusters)")
    axes[1].set_ylabel("Silueta")
    axes[1].set_title(f"{title_prefix} — Coeficiente de silueta")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    return inertias, silhouettes


def plot_clusters_2d(X, labels, centers=None, title: str = "Clusters (PCA 2D)"):
    """Proyeciona los datos a 2D con PCA y pinta los clusters."""
    X = np.asarray(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(X2[:, 0], X2[:, 1], c=labels, cmap="viridis", alpha=0.7, edgecolor="k", linewidth=0.3)
    if centers is not None:
        c2 = pca.transform(centers)
        plt.scatter(c2[:, 0], c2[:, 1], c="red", marker="X", s=150, label="Centros")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.title(title)
    plt.colorbar(scatter, label="Cluster")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return X2


def plot_decision_boundary_2d(X, y, model, title: str = "Frontera de decisión (2D)"):
    """Dibuja la frontera de decisión de un clasificador sobre 2 features."""
    X = np.asarray(X)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolor="k", linewidth=0.4)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.colorbar(scatter, label="Clase")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
