# Instalacion En Otro PC

Esta guia deja `fenrir` listo en un equipo nuevo sin depender de rutas absolutas del notebook original.

## Opcion recomendada si lo publicas en GitHub

Si `fenrir` ya esta subido como repo independiente, puedes instalarlo como libreria sin clonar el proyecto completo:

```bash
pip install "fenrir @ git+https://github.com/<tu-usuario>/fenrir.git"
```

Si tambien quieres widgets y workbench:

```bash
pip install "fenrir[notebook] @ git+https://github.com/<tu-usuario>/fenrir.git"
```

Valida la importacion:

```bash
python -c "from fenrir import Fenrir; print(Fenrir.__name__)"
```

## Opcion recomendada con conda

1. Copia el repositorio completo al otro PC.
2. Abre una terminal en esa carpeta.
3. Ejecuta:

```bash
conda env create -f environment.yml
conda activate fenrir
```

4. Valida la instalacion:

```bash
python examples/smoke_test.py
```

## Opcion con pip si clonas o copias el repo

1. Crea un entorno virtual.
2. Entra en la raiz del repositorio (donde esta `pyproject.toml`).
3. Ejecuta:

```bash
pip install -r requirements.txt
pip install -e .
```

4. Valida la importacion:

```bash
python -c "from fenrir import Fenrir; print(Fenrir.__name__)"
```

## Si luego quieres notebooks interactivos

La UI interactiva ya forma parte del paquete. Para usar widgets y workbench instala el extra `notebook`:

```bash
pip install -e .[notebook]
```

Si estas instalando desde GitHub:

```bash
pip install "fenrir[notebook] @ git+https://github.com/<tu-usuario>/fenrir.git"
```

## Nota sobre notebooks portables

- El notebook del repo `examples/fenrir_end_to_end_demo.ipynb` ya intenta importar la libreria instalada primero y luego el repo local.
- La plantilla `examples/fenrir_dataframe_template.ipynb` esta pensada para conectarse al `DataFrame` real del usuario, no a un dataset fijo del repositorio.
- Si usas un notebook propio con CSV externos, evita rutas duras como `c:\Users\...`.
- Coloca esos CSV junto al notebook o dentro de `examples/` o `data/` y resuelvelos con rutas relativas.

## Checklist minimo

- Python 3.10 o superior.
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`.
- Acceso al repo completo si quieres editar codigo o usar los ejemplos.
- Ejecutar `examples/smoke_test.py` tras instalar cuando tengas el repo local.
