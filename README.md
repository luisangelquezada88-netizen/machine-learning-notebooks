# Machine Learning – Cuadernos reproducibles

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="Licencia MIT">
  </a>
  <a href="https://www.linkedin.com/in/luis-angel-quezada-aguirre-291333268/">
    <img src="https://img.shields.io/badge/LinkedIn-Luis_Angel_Quezada-0077b5?logo=linkedin" alt="LinkedIn">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/python-3.14-blue?logo=python" alt="Python versión">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn" alt="scikit-learn">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/jupyter-notebook-orange?logo=jupyter" alt="Jupyter Notebook">
  </a>
</p>

> **Del dato crudo al modelo que decide.** Una colección de cuadernos de Machine Learning que no se queda en la teoría: cada notebook parte de un problema real, lo limpia, lo explora, lo modela y lo evalúa con métricas que importan.

Este repositorio nace de una convicción sencilla: **un modelo solo demuestra su valor cuando otro puede reproducirlo**. Por eso cada cuaderno aquí es una historia completa y ejecutable de principio a fin — desde la descarga del dataset hasta la interpretación de los resultados —, escrito con código limpio, sin atajos y sin dependencias ocultas.

No encontrarás celdas mágicas ni resultados inventados. Encontrarás **regresión, clasificación y clustering** aplicados a datos reales de medio ambiente, salud, finanzas, transporte y marketing, con la misma disciplina con la que se trabaja en producción: validación cruzada, métricas comparativas, matrices de confusión, importancia de variables y visualizaciones que explican *por qué* el modelo funciona (o no).

---

## ¿Por qué este repositorio?

- **Reproducible de verdad.** Un entorno aislado, dependencias fijadas y datasets con caché local. Clonas, instalas y ejecutas: los resultados se repiten.
- **Código que se lee.** Cada decisión —preprocesamiento, escalado, elección de hiperparámetros— está justificada en el propio cuaderno.
- **Problemas con contexto.** No son datasets de juguete: emisiones de CO₂, churn telefónico, cáncer, fraude bancario, propinas de taxi, segmentación de clientes.
- **De la métrica a la decisión.** No basta con entrenar: se compara, se interpreta y se concluye qué modelo conviene y por qué.

---

## Inventario de cuadernos y modelos

| Modelo | Cuaderno | Dominio | Métricas | Qué resuelve |
|--------|----------|---------|----------|--------------|
| Regresión lineal simple | `01-regresion/simple-linear-regression-co2.ipynb` | Medio ambiente | MAE, MSE, R², CV 5-fold | Estima emisiones de CO₂ a partir del tamaño del motor |
| Regresión lineal múltiple | `01-regresion/multiple-linear-regression-co2.ipynb` | Medio ambiente | MAE, MSE, R², CV 5-fold | Predice CO₂ combinando varias variables de consumo |
| Árbol de regresión | `01-regresion/regression-tree-taxi-tip.ipynb` | Transporte / Finanzas | RMSE, R², importancia de variables | Anticipa la propina de un viaje en taxi (NYC TLC) |
| Regresión con dataset Cloud | `01-regresion/regresion-cloud-seeds.ipynb` | Ciencia de datos | R², MAE, RMSE | Modela precipitaciones en experimentos de siembra de nubes |
| Regresión logística | `02-clasificacion/logistic-regression-churn.ipynb` | Finanzas / Telco | Exactitud, matriz de confusión, ROC-AUC | Detecta clientes con riesgo de abandono (churn) |
| Árbol de decisión | `02-clasificacion/decision-trees-drug.ipynb` | Salud | Exactitud, matriz de confusión | Recomienda un fármaco según el perfil del paciente |
| K-Vecinos Cercanos (KNN) | `02-clasificacion/k-nearest-neighbors-custcat.ipynb` | Finanzas / Telco | Exactitud, k óptimo | Clasifica el segmento de un cliente |
| SVM | `02-clasificacion/svm-cancer.ipynb` | Salud (cáncer) | Exactitud, matriz de confusión | Distingue células benignas de malignas |
| Árbol de decisión (fraude) | `02-clasificacion/decision-tree-credit-card-fraud.ipynb` | Finanzas (fraude) | Exactitud, matriz de confusión, ROC-AUC | Identifica transacciones fraudulentas |
| Softmax OvR / OvO | `02-clasificacion/multiclass-softmax-ovr-ovo.ipynb` | Multiclase | Exactitud por estrategia | Compara estrategias de clasificación multiclase |
| Clasificación con dataset Phoneme | `02-clasificacion/phoneme-clasificacion-models.ipynb` | Ciencia de datos | Exactitud, F1, ROC-AUC | Distingue fonemas nasales de orales con cuatro modelos |
| K-Means | `03-clustering/k-means-customer-segmentation.ipynb` | Marketing | Inercia, silueta, n.º de clusters | Segmenta clientes para decisiones comerciales |
| Clustering con dataset Cloud | `03-clustering/cloud-clustering.ipynb` | Ciencia de datos | Inercia, silueta | Descubre estructuras ocultas en datos de siembra de nubes |

> ⚠️ Cuaderno histórico duplicado (no usar): `docs/deprecated/regression-trees-DUPLICADO-logistic-churn.ipynb`

---

## Estructura

```
├── notebooks/
│   ├── 01-regresion/
│   │   ├── simple-linear-regression-co2.ipynb
│   │   ├── multiple-linear-regression-co2.ipynb
│   │   ├── regression-tree-taxi-tip.ipynb
│   │   └── regresion-cloud-seeds.ipynb
│   ├── 02-clasificacion/
│   │   ├── logistic-regression-churn.ipynb
│   │   ├── decision-trees-drug.ipynb
│   │   ├── k-nearest-neighbors-custcat.ipynb
│   │   ├── svm-cancer.ipynb
│   │   ├── decision-tree-credit-card-fraud.ipynb
│   │   ├── multiclass-softmax-ovr-ovo.ipynb
│   │   └── phoneme-clasificacion-models.ipynb
│   └── 03-clustering/
│       ├── k-means-customer-segmentation.ipynb
│       └── cloud-clustering.ipynb
├── src/utils.py              # set_seed, download_if_missing, load_csv_cached, plot_confusion_matrix + helpers de visualización
├── data/README.md            # fuentes y regla de caché (los CSV no se versionan)
├── docs/deprecated/          # duplicado histórico (no usar)
├── requirements.txt
└── environment.yml
```

## Environment (aislado, recomendado)

Trabaja siempre en un entorno aislado: así lo que ejecutas en tu máquina es exactamente lo que se ejecuta en cualquier otra.

```bash
# Opción A — venv (ligero, recomendado en Windows)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Opción B — conda
conda env create -f environment.yml
conda activate ml-notebooks
```

`.venv/` y `environment/` están ignorados por `.gitignore`: no se suben a GitHub, solo se versionan `requirements.txt` y `environment.yml`.

## Reproducir en 3 pasos

```bash
pip install -r requirements.txt
# o: conda env create -f environment.yml && conda activate ml-notebooks

jupyter lab
# Abrir cualquier notebook en notebooks/*/ y: Kernel > Restart & Run All
```

Validar que todos los cuadernos son JSON válido:

```bash
python -c "import json,glob; [json.load(open(f,encoding='utf-8')) for f in glob.glob('notebooks/**/*.ipynb',recursive=True)]; print('JSON OK')"
```

## Créditos

- Labs base IBM SkillsNetwork ML0101EN (regresión, clasificación, clustering).
- Proyecto UNAD curso 203008067, grupo 33 – Luis Angel Quezada Aguirre (Fases 2 y 3).
- Licencia MIT. Ver `LICENSE`.
