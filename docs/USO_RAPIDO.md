# Uso Rapido

Fenrir trabaja con cualquier `pandas.DataFrame` del usuario. Los ejemplos de esta guia son solo eso: ejemplos minimos para enseñar el flujo de uso.

## Caso minimo

```python
import pandas as pd
from fenrir import Fenrir

df = pd.DataFrame(
    {
        "ingresos": [100, 120, 130, 400, 420, 410],
        "margen": [10, 12, 11, 40, 38, 42],
        "tickets": [1, 2, 1, 9, 8, 10],
    }
)

fenrir = Fenrir(df, variance_threshold=0.85, cluster_range=range(2, 4))
fenrir.fit()

print(fenrir.best_configuration())
print(fenrir.metric_report())
print(fenrir.cluster_size_report())
```

## Con variable objetivo

```python
import pandas as pd
from fenrir import Fenrir

df = pd.DataFrame(
    {
        "x1": [1, 2, 3, 20, 21, 22],
        "x2": [1, 1, 2, 9, 10, 9],
        "segmento": ["a", "a", "a", "b", "b", "b"],
    }
)

fenrir = Fenrir(df, target="segmento", variance_threshold=0.8, cluster_range=range(2, 4))
fenrir.fit()

print(fenrir.metric_report())
print(fenrir.cluster_target_report())
print(fenrir.evaluate_holdout())
```

## Desde un bundle de Bahamut

```python
from fenrir import Fenrir

modelo = Fenrir.from_bahamut(segmentos, fit=True)
print(modelo.bahamut_split_report())
```

Ajusta solo con `train` y puntua `validation` y `test` sin reajustar. Ver el README
para las opciones (`target=`, `df=`, `include_train=`).

## Metodos clave

- `fit()`
- `fit_predict()`
- `predict(data)`
- `summary(top_n=10)`
- `metric_report()`
- `cluster_size_report()`
- `cluster_groupby()`
- `evaluate_holdout()`
- `from_bahamut(segmentos)` y `evaluate_bahamut_splits()`
- `evaluate_stability()`
- `analysis_report()`
