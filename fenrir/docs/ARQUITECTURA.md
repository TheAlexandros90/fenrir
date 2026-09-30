# Arquitectura

## Objetivo

Separar el nucleo analitico reutilizable del notebook interactivo.

## Estado actual

- El paquete `fenrir` contiene la clase `Fenrir`, el core analitico y la capa interactiva en `interactive.py`.
- El notebook puede reutilizar el paquete directamente sin redefinir widgets ni exportadores en celdas.

## Frontera con Bahamut

Bahamut es la fuente de verdad de los cortes `train` / `validation` / `test`.
`Fenrir.from_bahamut(segmentos)` consume ese bundle: ajusta con `train` y evalua
los demas bloques proyectandolos con el escalado y el PCA ya ajustados. Fenrir no
redefine la segmentacion ni la discute; solo la respeta.

Eso deja dos evaluaciones con proposito distinto, y conviene no confundirlas:

- `evaluate_bahamut_splits()` responde "como se comporta este modelo en los bloques
  que decidio Bahamut". No reajusta nada.
- `evaluate_holdout()` responde "como de estable es la busqueda". Reajusta el
  pipeline completo en particiones aleatorias propias.

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

## Siguiente paso recomendado

Con la frontera con Bahamut ya cerrada, lo que queda es aplanar el repositorio: el
paquete vive hoy dos niveles por debajo de la raiz del clon, lo que hace que las
instrucciones de instalacion solo funcionen desde una de las dos carpetas llamadas
`fenrir`.
