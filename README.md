# Machine Learning – Cuadernos reproducibles

<p align="center">
  <a href="https://github.com/luis-quezada/ml-notebooks/actions/workflows/notebooks-ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/luis-quezada/ml-notebooks/notebooks-ci.yml?branch=main&label=CI%20notebooks&color=green" alt="CI notebooks">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="Licencia MIT">
  </a>
  <a href="https://www.linkedin.com/in/luis-quezada/">
    <img src="https://img.shields.io/badge/LinkedIn-Luis_Quezada-0077b5?logo=linkedin" alt="LinkedIn">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python" alt="Python versión">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn" alt="scikit-learn">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/jupyter-notebook-orange?logo=jupyter" alt="Jupyter Notebook">
  </a>
</p>

El presente repositorio contiene varios cuadernos de Python `.ipynb` con diferentes algoritmos de aprendizaje de máquina (**Machine Learning**). Son cuadernos con código limpio, listo para reproducir y reutilizar, provenientes de cursos y materiales de estudio universitario.

El repositorio abarca diferentes técnicas, herramientas y necesidades del aprendizaje de máquina. Contiene algoritmos de **regresión, clasificación y agrupamiento (clustering)**, así como la revisión de hiperparámetros, métricas para la evaluación comparativa de modelos y visualizaciones pertinentes.

Pueden usarse para diferentes campos del conocimiento. Los datos analizados en los cuadernos abarcan **medio ambiente, medicina, finanzas, información geográfica** y demás disciplinas, entendiendo a los datos como pieza transversal a cualquier sector productivo y académico.

---

## Inventario de cuadernos y modelos

| Modelo | Cuaderno | Tema | Indicadores | Resultado |
|--------|----------|------|-------------|-----------|
| Regresión lineal simple | `01-regresion/simple-linear-regression-co2.ipynb` | Ambiental (CO₂) | MAE, MSE, R², CV 5-fold | Predice emisiones CO₂ a partir del tamaño del motor |
| Regresión lineal múltiple | `01-regresion/multiple-linear-regression-co2.ipynb` | Ambiental (CO₂) | MAE, MSE, R², CV 5-fold | Predice CO₂ usando varias variables de consumo |
| Árbol de regresión | `01-regresion/regression-tree-taxi-tip.ipynb` | Finanzas / Transporte | RMSE, R², importancia de features | Predice propina de taxi (NYC TLC) |
| Regresión logística | `02-clasificacion/logistic-regression-churn.ipynb` | Finanzas / Telco | Acuracay, matriz de confusión, ROC-AUC | Predice churn de clientes |
| Árbol de decisión | `02-clasificacion/decision-trees-drug.ipynb` | Salud | Acuracay, matriz de confusión | Recomienda fármaco según perfil del paciente |
| K-Vecinos Cercanos (KNN) | `02-clasificacion/k-nearest-neighbors-custcat.ipynb` | Finanzas / Telco | Acuracay, k óptimo | Clasifica segmento de cliente |
| SVM | `02-clasificacion/svm-cancer.ipynb` | Salud (cáncer) | Acuracay, matriz de confusión | Clasifica células benignas/malignas |
| Árbol de decisión (fraude) | `02-clasificacion/decision-tree-credit-card-fraud.ipynb` | Finanzas (fraude) | Acuracay, matriz de confusión | Detecta transacciones fraudulentas |
| Softmax OvR / OvO | `02-clasificacion/multiclass-softmax-ovr-ovo.ipynb` | Multiclase | Acuracay por estrategia | Clasificación multiclase |
| K-Means | `03-clustering/k-means-customer-segmentation.ipynb` | Marketing | Inercia, silueta, n.º de clusters | Segmentación de clientes |
| Regresión + Clasificación (OpenML) | `04-proyecto-unad/fase2-regresion-clasificacion-cloud-phoneme.ipynb` | Ciencia de datos | R², acuracay | Proyecto supervisado UNAD (Cloud / Phoneme) |
| Clustering no supervisado (OpenML) | `04-proyecto-unad/fase3-clustering.ipynb` | Ciencia de datos | Inercia, silueta | Proyecto no supervisado UNAD |

> Cuaderno histórico duplicado (no usar): `docs/deprecated/regression-trees-DUPLICADO-logistic-churn.ipynb`

---

## Estructura

```
├── notebooks/
│   ├── 01-regresion/
│   │   ├── simple-linear-regression-co2.ipynb
│   │   ├── multiple-linear-regression-co2.ipynb
│   │   └── regression-tree-taxi-tip.ipynb
│   ├── 02-clasificacion/
│   │   ├── logistic-regression-churn.ipynb
│   │   ├── decision-trees-drug.ipynb
│   │   ├── k-nearest-neighbors-custcat.ipynb
│   │   ├── svm-cancer.ipynb
│   │   ├── decision-tree-credit-card-fraud.ipynb
│   │   └── multiclass-softmax-ovr-ovo.ipynb
│   ├── 03-clustering/
│   │   └── k-means-customer-segmentation.ipynb
│   └── 04-proyecto-unad/
│       ├── fase2-regresion-clasificacion-cloud-phoneme.ipynb
│       └── fase3-clustering.ipynb
├── src/utils.py              # set_seed, download_if_missing, load_csv_cached, plot_confusion_matrix + helpers de visualización
├── data/README.md            # fuentes y regla de cache (los CSV no se versionan)
├── docs/
│   ├── MEJORAS-POR-NOTEBOOK.md
│   └── deprecated/           # duplicado histórico (no usar)
├── requirements.txt
├── environment.yml
└── .github/workflows/notebooks-ci.yml
```

## Environment (aislado, recomendado)

Sí: trabaja siempre en un environment aislado para que lo que ejecutes aquí sea lo mismo que se ejecute en GitHub/CI.

```bash
# Opción A — venv (ligero, recomendado en este equipo Windows)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Opción B — conda
conda env create -f environment.yml
conda activate ml-notebooks
```

`.venv/` y `environment/` ya están ignorados por `.gitignore`: no se suben a GitHub, solo se versionan `requirements.txt` y `environment.yml`.

## Reproducir en 3 pasos

```bash
pip install -r requirements.txt
# o: conda env create -f environment.yml && conda activate ml-notebooks

jupyter lab
# Abrir cualquier notebook en notebooks/*/ y: Kernel > Restart & Run All
```

Validar JSON + ausencia de `!pip install`:

```bash
python -c "import json,glob; [json.load(open(f,encoding='utf-8')) for f in glob.glob('notebooks/**/*.ipynb',recursive=True)]; print('JSON OK')"
python -m nbconvert --to notebook --execute "notebooks/02-clasificacion/decision-trees-drug.ipynb" --output /tmp/test.ipynb --allow-errors
```

## Créditos

- Labs base IBM SkillsNetwork ML0101EN (regresión, clasificación, clustering).
- Proyecto UNAD curso 203008067, grupo 33 – Luis Quezada (Fase 2 y 3).
- Licencia MIT. Ver `LICENSE`.
