# fenrir

`fenrir` es un paquete de Python para clustering guiado por PCA con foco en dos cosas: reducir codigo repetitivo y mantener calidad analitica.

## Que incluye ahora mismo

- Trabajo sobre cualquier `pandas.DataFrame` que entregue el usuario.
- Ajuste automatico de PCA y comparativa de escalados.
- Busqueda del mejor clustering entre `kmeans`, `gmm` y `agglomerative`.
- Metricas internas y externas cuando existe variable objetivo.
- Holdout repetido y estabilidad por remuestreo.
- Busqueda de subconjuntos compactos con variables influyentes.
- Graficos base de PCA y clusters.
- Capa interactiva empaquetada con tablas de PCA y clustering.
- Workbench con presets, persistencia JSON, comparador y exportacion ejecutiva a Excel.
- Traspaso directo desde un bundle de Bahamut, respetando sus cortes.

## Estructura

El repositorio es a la vez la raiz del proyecto y el paquete: `pyproject.toml`
convive con los modulos, asi que todos los comandos de instalacion se lanzan
desde el directorio que se clona, sin subcarpetas intermedias.

```text
fenrir/
|-- pyproject.toml
|-- README.md
|-- __init__.py
|-- core.py
|-- interactive.py
|-- _notebook.py
|-- docs/
|-- examples/
`-- tests/
```

- `core.py`: implementacion principal de la clase `Fenrir`.
- `interactive.py`: capa interactiva empaquetada sobre el mismo motor analitico.
- `__init__.py`: punto de importacion limpio.
- `examples/smoke_test.py`: prueba minima para validar que la instalacion funciona.
- `examples/fenrir_end_to_end_demo.ipynb`: demo completa en notebook con leaderboard, perfiles, holdout, estabilidad y exportacion ejecutiva. Intenta importar primero la libreria instalada y, si no existe, usa el repo local.
- `examples/fenrir_dataframe_template.ipynb`: plantilla generica para conectar Fenrir con el `DataFrame` real del usuario o con un ejemplo minimo interno.
- `docs/`: documentacion de instalacion, uso y arquitectura.
- `tests/`: pruebas automatizadas con `pytest`.

## Instalacion rapida

### Opcion 1. pip editable local

Desde la raiz del repositorio:

```bash
pip install -e .
```

Para usar widgets y workbench:

```bash
pip install -e .[notebook]
```

### Opcion 2. instalar desde GitHub

Si publicas `fenrir` como repo independiente en GitHub, en cualquier PC puedes instalarlo sin clonar todo el proyecto:

```bash
pip install "fenrir @ git+https://github.com/<tu-usuario>/fenrir.git"
```

Para usar widgets y workbench:

```bash
pip install "fenrir[notebook] @ git+https://github.com/<tu-usuario>/fenrir.git"
```

### Opcion 3. conda

Desde la raiz del repositorio:

```bash
conda env create -f environment.yml
conda activate fenrir
```

## Uso en otro PC

Hay dos formas normales de mover Fenrir a otra maquina:

1. Si solo quieres consumir la libreria, instala directamente desde GitHub con `pip install "fenrir @ git+https://github.com/<tu-usuario>/fenrir.git"`.
2. Si tambien quieres editar codigo, ejecutar tests o abrir los notebooks del repo, clona el proyecto y usa `pip install -e .[notebook]` o `conda env create -f environment.yml`.
3. Valida la instalacion con `python examples/smoke_test.py` o con `python -c "from fenrir import Fenrir; print(Fenrir.__name__)"`.

## Uso rapido

```python
import pandas as pd
from fenrir import Fenrir

df = pd.DataFrame(
    {
        "x1": [1, 2, 3, 4, 5, 6],
        "x2": [1, 1, 2, 2, 3, 3],
        "x3": [10, 11, 10, 11, 30, 31],
    }
)

model = Fenrir(df, variance_threshold=0.8, cluster_range=range(2, 4))
labels = model.fit_predict()
print(labels)
print(model.best_configuration())
```

Fenrir no depende de un dataset concreto. Recibe el `DataFrame` que el usuario quiera analizar y, si existe, una columna objetivo opcional para enriquecer la lectura de clusters.

## Notebook de referencia

Si quieres un flujo mas guiado y orientado a explotacion de resultados, revisa:

- `examples/fenrir_end_to_end_demo.ipynb`
- `examples/fenrir_dataframe_template.ipynb`

`examples/fenrir_end_to_end_demo.ipynb` es la demo mas estable para clonar y ejecutar tal cual, porque usa datos sinteticos y no depende de ficheros externos.

`examples/fenrir_dataframe_template.ipynb` es la plantilla pensada para pegar o asignar el `DataFrame` real del usuario y ajustar `target` y `exclude_columns` a sus columnas.

## Traspaso desde Bahamut

Si los cortes ya los decidio Bahamut, Fenrir puede respetarlos en vez de inventarse
sus propias particiones:

```python
from bahamut import BahamutSplit
from fenrir import Fenrir

segmentos = (
    BahamutSplit(df)
    .definir_problema("clasificacion")
    .definir_objetivo("segmento")
    .definir_estratificacion(modo="target")
    .configurar_split(test_size=0.2, validation_size=0.2, shuffle=True, random_state=11)
    .ejecutar_segmentacion()
)

modelo = Fenrir.from_bahamut(segmentos, fit=True)
print(modelo.bahamut_split_report())
```

`from_bahamut()` ajusta solo con el bloque `train`. `evaluate_bahamut_splits()`
proyecta `validation` y `test` con el mismo escalado y el mismo PCA del ajuste y
les asigna clusters con el modelo base: no reajusta nada. Es la lectura fuera de
muestra que corresponde a los cortes decididos aguas arriba.

La diferencia con `evaluate_holdout()` importa: ese metodo se inventa particiones
aleatorias y reajusta el pipeline entero en cada una, lo que responde a otra
pregunta ("como de estable es la busqueda") y no a esta ("como se comporta *este*
modelo en los bloques que aparte").

Detalles utiles:

- La columna objetivo se deduce de `y_train` cuando trae una sola columna; si trae
  varias, hay que elegir con `target=`.
- `df=dataframe_original` rehidrata cada bloque por indices y recupera las columnas
  que Bahamut dejo fuera de `feature_cols`. El indice de `df` debe ser unico.
  Esas columnas quedan disponibles para consultar, pero las predictoras siguen
  siendo las de `X_train`; puedes cambiarlas de forma explicita con `features=`.
- Funciona tambien con `agglomerative`, que no admite `predict()`: los bloques no
  vistos se asignan al centroide mas cercano del ajuste.

## Workbench interactivo

```python
from fenrir import Fenrir

model = Fenrir(df, variance_threshold=0.8, cluster_range=range(2, 4)).fit()
model.interactive_workbench()
```

## Alcance actual

El paquete contiene tanto el nucleo analitico como la capa interactiva. El notebook puede seguir usandose como laboratorio, pero la UI principal ya no depende de celdas parcheadas.

## Notebooks portables

- El patron recomendado es importar `fenrir` instalado primero y usar el repo local solo como fallback.
- Evita rutas fijas como `c:\Users\...` para localizar el paquete o datos auxiliares.
- Si el notebook consume ficheros externos, guardalos junto al notebook o dentro de `examples/` o `data/` del proyecto y resuelvelos con rutas relativas.

## Documentacion adicional

- `docs/INSTALACION_OTRO_PC.md`
- `docs/USO_RAPIDO.md`
- `docs/ARQUITECTURA.md`
