# Arquitectura

## Objetivo

Separar el nucleo analitico reutilizable del notebook interactivo.

## Estado actual

- El paquete `fenrir` contiene la clase `Fenrir`, el core analitico y la capa interactiva en `interactive.py`.
- El notebook puede reutilizar el paquete directamente sin redefinir widgets ni exportadores en celdas.

## Responsabilidad del core

- Preparacion de datos.
- Codificacion de categoricas de baja cardinalidad.
- Imputacion.
- PCA y diagnosticos.
- Seleccion de mejor configuracion de clustering.
- Evaluacion holdout y estabilidad.
- Reportes tabulares.
- Graficos base.

## Responsabilidad de interactive.py

- `ipywidgets`.
- Persistencia interactiva de presets.
- Exportaciones manuales desde interfaz.
- Comparaciones visuales y workbench de usuario.
- Publicacion opcional de bindings en el espacio de usuario de IPython para mantener un flujo low-code.

## Responsabilidad del notebook

- Ser entorno de exploracion, demos y validacion visual.
- Servir como escaparate de casos de uso del paquete.

## Siguiente extraccion recomendada

El siguiente paso natural ya no es mover widgets fuera del notebook, sino endurecer la superficie publica con mas tests y ejemplos de uso orientados a escenarios reales.
