# data/

Los datasets **no se versionan** en GitHub (ver `.gitignore`).

- `data/raw/` — cache local de los CSV descargados por los notebooks.
- `data/processed/` — salidas pequeñas derivadas (muestras).

## Fuentes

| Dataset | URL base | Usado en |
|---|---|---|
| ChurnData.csv | `IBMDeveloperSkillsNetwork-ML0101EN/.../Module 3/data/ChurnData.csv` | regresión logística churn |
| cell_samples.csv | `.../Module 3/data/cell_samples.csv` | SVM cáncer |
| drug200.csv | `.../Module 3/data/drug200.csv` | árboles decisión |
| teleCust1000t.csv | `.../Module 3/data/teleCust1000t.csv` | KNN |
| creditcard.csv | `.../Module 3/data/creditcard.csv` | fraud detection |
| FuelConsumptionCo2.csv | `.../Module 2/data/FuelConsumptionCo2.csv` | regresión lineal |
| Cust_Segmentation.csv | `.../Module 4/data/Cust_Segmentation.csv` | K-Means |
| yellow_tripdata_2019-06.csv | NYC TLC (~1 GB, usar muestra) | taxi tip |
| OpenML ID 210 (Cloud), ID 1489 (Phoneme) | `fetch_openml` | proyecto UNAD Fase 2/3 |

## Regla

```python
from src.utils import load_csv_cached
df = load_csv_cached(URL, "ChurnData.csv")
```

Si no hay internet, coloca el CSV manualmente en `data/raw/` con el mismo nombre.
Para el dataset de taxis, **no descargues el CSV completo**: usa `nrows=100_000` o una muestra en `data/raw/yellow_tripdata_sample.csv`.
