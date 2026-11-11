# Machine Learning – Cuadernos reproducibles

El presente repositorio contiene varios cuadernos de Python `.ipynb` con diferentes algoritmos de aprendizaje de máquina (Machine Learning). Son cuadernos con código limpio, listo para reproducir y reutilizar, provenientes de cursos y materiales de estudio universitario.

El repositorio abarca diferentes técnicas, herramientas y necesidades del aprendizaje de máquina. Contiene algoritmos de regresión, clasificación y agrupamiento (clustering), así como la revisión de hiperparámetros, métricas para la evaluación comparativa de modelos y visualizaciones pertinentes.

Pueden usarse para diferentes campos del conocimiento. Los datos analizados en los cuadernos abarcan medio ambiente, medicina, finanzas, información geográfica y demás disciplinas, ententiendo a los datos como pieza transversal a cualquier sector productivo y académico.

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
├── src/utils.py              # set_seed, download_if_missing, load_csv_cached, plot_confusion_matrix
├── data/README.md            # fuentes y regla de cache (los CSV no se versionan)
├── docs/
│   ├── MEJORAS-POR-NOTEBOOK.md
│   └── deprecated/           # duplicado historico (no usar)
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
