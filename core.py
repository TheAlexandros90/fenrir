from __future__ import annotations

import importlib
import numpy as np
import pandas as pd
from typing import Any, Literal, Mapping, overload
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def _mode_or_nan(series: pd.Series):
    mode = series.mode(dropna=True)
    if mode.empty:
        return np.nan
    return mode.iloc[0]


def _interpret_metric(metric: str, value) -> str:
    if value is None or pd.isna(value):
        return "Sin valor disponible"

    value = float(value)
    if metric == "explained_variance":
        if value >= 0.9:
            return "Muy fuerte"
        if value >= 0.85:
            return "Fuerte"
        if value >= 0.7:
            return "Aceptable"
        return "Baja para un PCA orientado a clustering"
    if metric == "variance_first_two":
        if value >= 0.6:
            return "Muy util para visualizacion 2D"
        if value >= 0.45:
            return "Visualizacion razonable"
        return "La proyeccion 2D pierde bastante informacion"
    if metric == "silhouette":
        if value >= 0.7:
            return "Separacion excelente"
        if value >= 0.5:
            return "Separacion buena"
        if value >= 0.25:
            return "Separacion moderada"
        return "Separacion floja"
    if metric == "davies_bouldin":
        if value < 0.6:
            return "Muy bueno"
        if value < 1.0:
            return "Bueno"
        if value < 1.5:
            return "Regular"
        return "Flojo"
    if metric == "calinski_harabasz":
        return "Solo interpretable en comparacion con otras configuraciones del mismo dataset"
    if metric in {"ari", "nmi", "v_measure"}:
        if value >= 0.75:
            return "Alineacion muy fuerte con la objetivo"
        if value >= 0.5:
            return "Alineacion buena"
        if value >= 0.3:
            return "Alineacion apreciable"
        return "Alineacion debil"
    if metric == "n_clusters":
        return "Valido si tambien es estable en holdout y sigue siendo interpretable"
    if metric == "n_components":
        return "Mejor cuanto mas pequeno sea, si mantiene el rendimiento"
    return "Interpretacion orientativa no definida"


def _interpret_stability(score) -> str:
    if pd.isna(score):
        return "Sin evidencia suficiente"
    if score >= 0.75:
        return "Muy estable"
    if score >= 0.50:
        return "Estable"
    if score >= 0.30:
        return "Aceptable con cautela"
    return "Fragil"


def _require_matplotlib():
    try:
        plt_module = importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise ImportError(
            "matplotlib es necesario para las funciones de visualizacion. "
            "Instalalo con 'pip install matplotlib' o usa las salidas tabulares de Fenrir."
        ) from exc
    return plt_module


def _algorithm_label(name: str) -> str:
    labels = {
        "kmeans": "K-Means",
        "gmm": "Gaussian Mixture",
        "agglomerative": "Agglomerative",
    }
    return labels.get(str(name), str(name))


def _metric_label(metric: str) -> str:
    labels = {
        "explained_variance": "Varianza explicada PCA",
        "variance_first_two": "Varianza acumulada en PC1-PC2",
        "silhouette": "Silhouette",
        "calinski_harabasz": "Calinski-Harabasz",
        "davies_bouldin": "Davies-Bouldin",
        "ari": "Adjusted Rand Index",
        "nmi": "Normalized Mutual Information",
        "v_measure": "V-Measure",
        "n_clusters": "Numero de clusters",
        "n_components": "Componentes PCA",
    }
    return labels.get(str(metric), str(metric))


def _build_configuration_label(row: pd.Series) -> str:
    algorithm = _algorithm_label(row.get("algorithm", ""))
    clusters_value = row.get("n_clusters", np.nan)
    components_value = row.get("n_components", np.nan)
    if pd.isna(clusters_value) or pd.isna(components_value):
        return algorithm
    try:
        clusters = int(clusters_value)
        components = int(components_value)
    except (TypeError, ValueError):
        return algorithm
    return f"{algorithm} | k={clusters} | PCA={components}"


def _safe_round(value, digits: int = 4):
    if value is None or pd.isna(value):
        return np.nan
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def _round_existing_columns(frame: pd.DataFrame, columns, digits: int = 4) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].apply(lambda value: _safe_round(value, digits=digits))
    return frame


def _format_metric_value(metric: str, value) -> str:
    if value is None or pd.isna(value):
        return "Sin valor disponible"
    if metric in {"explained_variance", "variance_first_two"}:
        return f"{float(value):.1%}"
    if metric in {"n_clusters", "n_components"}:
        return str(int(value))
    return f"{float(value):.4f}"


def _interpret_cluster_share(share) -> str:
    if share is None or pd.isna(share):
        return "Sin referencia"
    share = float(share)
    if share < 0.05:
        return "Muy pequeno"
    if share < 0.10:
        return "Pequeno"
    if share < 0.25:
        return "Medio"
    if share < 0.45:
        return "Grande"
    return "Muy dominante"


def _interpret_target_concentration(share) -> str:
    if share is None or pd.isna(share):
        return "Sin lectura"
    share = float(share)
    if share >= 0.85:
        return "Muy concentrado"
    if share >= 0.70:
        return "Concentracion clara"
    if share >= 0.55:
        return "Concentracion moderada"
    return "Mezcla amplia"


def _decorate_candidate_table(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.copy().reset_index(drop=True)
    if table.empty:
        return table

    table.insert(0, "rank", np.arange(1, len(table) + 1))
    if "algorithm" in table.columns:
        table["algorithm_label"] = table["algorithm"].map(_algorithm_label)
    if {"algorithm", "n_clusters", "n_components"}.issubset(table.columns):
        table["configuration"] = table.apply(_build_configuration_label, axis=1)
    if "explained_variance" in table.columns:
        table["explained_variance_reading"] = table["explained_variance"].map(
            lambda value: _interpret_metric("explained_variance", value)
        )
    if "variance_first_two" in table.columns:
        table["variance_first_two_reading"] = table["variance_first_two"].map(
            lambda value: _interpret_metric("variance_first_two", value)
        )
    if "silhouette" in table.columns:
        table["silhouette_reading"] = table["silhouette"].map(
            lambda value: _interpret_metric("silhouette", value)
        )
    if "davies_bouldin" in table.columns:
        table["davies_bouldin_reading"] = table["davies_bouldin"].map(
            lambda value: _interpret_metric("davies_bouldin", value)
        )

    table = _round_existing_columns(
        table,
        [
            "explained_variance",
            "variance_first_two",
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
        ],
    )

    preferred = [
        "rank",
        "scaler_name",
        "algorithm",
        "algorithm_label",
        "configuration",
        "n_clusters",
        "n_components",
        "explained_variance",
        "explained_variance_reading",
        "variance_first_two",
        "variance_first_two_reading",
        "silhouette",
        "silhouette_reading",
        "calinski_harabasz",
        "davies_bouldin",
        "davies_bouldin_reading",
    ]
    ordered = [column for column in preferred if column in table.columns]
    ordered += [column for column in table.columns if column not in ordered]
    return table.loc[:, ordered]


def _decorate_metric_table(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.copy().reset_index(drop=True)
    if table.empty:
        return table

    table["metric_label"] = table["metric"].map(_metric_label)
    table["value_display"] = table.apply(
        lambda row: _format_metric_value(row["metric"], row["value"]),
        axis=1,
    )
    table = _round_existing_columns(table, ["value"])

    preferred = [
        "metric",
        "metric_label",
        "value",
        "value_display",
        "direction",
        "acceptable",
        "interpretation",
        "notes",
    ]
    ordered = [column for column in preferred if column in table.columns]
    ordered += [column for column in table.columns if column not in ordered]
    return table.loc[:, ordered]


def _decorate_holdout_table(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.copy().reset_index(drop=True)
    if table.empty:
        return table

    table.insert(0, "rank", np.arange(1, len(table) + 1))
    if "algorithm" in table.columns:
        table["algorithm_label"] = table["algorithm"].map(_algorithm_label)
    if {"algorithm", "n_clusters", "n_components"}.issubset(table.columns):
        table["configuration"] = table.apply(_build_configuration_label, axis=1)
    if "explained_variance_mean" in table.columns:
        table["explained_variance_reading"] = table["explained_variance_mean"].map(
            lambda value: _interpret_metric("explained_variance", value)
        )
    if "test_silhouette_mean" in table.columns:
        table["holdout_silhouette_reading"] = table["test_silhouette_mean"].map(
            lambda value: _interpret_metric("silhouette", value)
        )
    if "test_davies_bouldin_mean" in table.columns:
        table["holdout_davies_bouldin_reading"] = table["test_davies_bouldin_mean"].map(
            lambda value: _interpret_metric("davies_bouldin", value)
        )
    if "test_v_measure_mean" in table.columns:
        table["target_alignment_reading"] = table["test_v_measure_mean"].map(
            lambda value: _interpret_metric("v_measure", value)
        )
    elif "test_nmi_mean" in table.columns:
        table["target_alignment_reading"] = table["test_nmi_mean"].map(
            lambda value: _interpret_metric("nmi", value)
        )

    table = _round_existing_columns(
        table,
        [
            "explained_variance_mean",
            "explained_variance_std",
            "variance_first_two_mean",
            "variance_first_two_std",
            "test_silhouette_mean",
            "test_silhouette_std",
            "test_calinski_harabasz_mean",
            "test_calinski_harabasz_std",
            "test_davies_bouldin_mean",
            "test_davies_bouldin_std",
            "test_ari_mean",
            "test_ari_std",
            "test_nmi_mean",
            "test_nmi_std",
            "test_v_measure_mean",
            "test_v_measure_std",
        ],
    )

    preferred = [
        "rank",
        "scaler_name",
        "algorithm",
        "algorithm_label",
        "configuration",
        "n_clusters",
        "n_components",
        "explained_variance_mean",
        "explained_variance_reading",
        "variance_first_two_mean",
        "test_silhouette_mean",
        "test_silhouette_std",
        "holdout_silhouette_reading",
        "test_davies_bouldin_mean",
        "test_davies_bouldin_std",
        "holdout_davies_bouldin_reading",
        "test_v_measure_mean",
        "test_nmi_mean",
        "test_ari_mean",
        "target_alignment_reading",
    ]
    ordered = [column for column in preferred if column in table.columns]
    ordered += [column for column in table.columns if column not in ordered]
    return table.loc[:, ordered]


def _decorate_stability_table(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.copy().reset_index(drop=True)
    if table.empty:
        return table

    if "algorithm" in table.columns:
        table["algorithm_label"] = table["algorithm"].map(_algorithm_label)
    if {"algorithm", "n_clusters", "n_components"}.issubset(table.columns):
        table["configuration"] = table.apply(_build_configuration_label, axis=1)
    if "ari_vs_base" in table.columns:
        table["ari_vs_base_reading"] = table["ari_vs_base"].map(_interpret_stability)
    if "silhouette" in table.columns:
        table["silhouette_reading"] = table["silhouette"].map(
            lambda value: _interpret_metric("silhouette", value)
        )
    same_config_columns = [
        column
        for column in ["same_scaler", "same_algorithm", "same_n_clusters", "same_n_components"]
        if column in table.columns
    ]
    if same_config_columns:
        table["same_configuration"] = table[same_config_columns].all(axis=1)
        table["configuration_status"] = np.where(
            table["same_configuration"],
            "Coincide con la configuracion base",
            "Se mueve respecto a la configuracion base",
        )

    table = _round_existing_columns(
        table,
        [
            "sample_fraction",
            "silhouette",
            "explained_variance",
            "davies_bouldin",
            "ari_vs_base",
            "nmi_vs_base",
            "v_measure_vs_base",
        ],
    )

    preferred = [
        "split",
        "sample_size",
        "sample_fraction",
        "scaler_name",
        "algorithm",
        "algorithm_label",
        "configuration",
        "n_clusters",
        "n_components",
        "silhouette",
        "silhouette_reading",
        "ari_vs_base",
        "ari_vs_base_reading",
        "nmi_vs_base",
        "v_measure_vs_base",
        "min_cluster_size",
        "same_configuration",
        "configuration_status",
    ]
    ordered = [column for column in preferred if column in table.columns]
    ordered += [column for column in table.columns if column not in ordered]
    return table.loc[:, ordered]


def _decorate_stability_summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.copy().reset_index(drop=True)
    if table.empty:
        return table

    if "ari_vs_base_mean" in table.columns:
        table["ari_vs_base_reading"] = table["ari_vs_base_mean"].map(_interpret_stability)
    if "silhouette_mean" in table.columns:
        table["silhouette_reading"] = table["silhouette_mean"].map(
            lambda value: _interpret_metric("silhouette", value)
        )

    table = _round_existing_columns(
        table,
        [
            "sample_fraction",
            "ari_vs_base_mean",
            "ari_vs_base_std",
            "nmi_vs_base_mean",
            "nmi_vs_base_std",
            "v_measure_vs_base_mean",
            "v_measure_vs_base_std",
            "same_scaler_rate",
            "same_algorithm_rate",
            "same_n_clusters_rate",
            "same_n_components_rate",
            "same_configuration_rate",
            "silhouette_mean",
            "min_cluster_size_mean",
        ],
    )
    return table


def _decorate_cluster_size_table(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.copy()
    if table.empty:
        return table
    if "cluster_share" in table.columns:
        table["cluster_balance"] = table["cluster_share"].map(_interpret_cluster_share)
    table = _round_existing_columns(table, ["cluster_share"])
    return table


def _decorate_cluster_groupby_table(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.copy()
    if table.empty:
        return table
    if "cluster_share" in table.columns:
        table["cluster_balance"] = table["cluster_share"].map(_interpret_cluster_share)
    table = _round_existing_columns(table, ["cluster_share"])
    return table


_BAHAMUT_SPLITS = (
    ("train", "X_train", "y_train", "indices_train"),
    ("validation", "X_validation", "y_validation", "indices_validation"),
    ("test", "X_test", "y_test", "indices_test"),
)


def _looks_like_bahamut_segments(segments) -> bool:
    if not isinstance(segments, Mapping):
        return False
    return "X_train" in segments and any(str(key).startswith("indices_") for key in segments)


def _coerce_bahamut_target(target_like, split_name: str):
    """Normaliza y_train / y_validation / y_test a DataFrame o None."""

    if target_like is None:
        return None
    if isinstance(target_like, pd.Series):
        if target_like.name is None:
            raise ValueError(
                f"El bloque '{split_name}' de Bahamut trae una serie objetivo sin nombre. "
                "Renombrala o indica target= explicitamente."
            )
        return target_like.to_frame()
    if isinstance(target_like, pd.DataFrame):
        return target_like.copy()
    raise TypeError(f"El bloque '{split_name}' de Bahamut trae un objetivo de tipo no soportado: {type(target_like)!r}")


def _resolve_bahamut_target_name(segments, target=None) -> str | None:
    """Decide que columna hace de objetivo al leer un bundle de Bahamut."""

    if target is not None:
        return str(target)

    target_frame = _coerce_bahamut_target(segments.get("y_train"), "train")
    if target_frame is None or target_frame.shape[1] == 0:
        return None
    if target_frame.shape[1] == 1:
        return str(target_frame.columns[0])

    raise ValueError(
        "El bundle de Bahamut trae varias columnas objetivo "
        f"({list(target_frame.columns)}). Fenrir perfila clusters contra una sola: "
        "indica cual con target=."
    )


def _bahamut_frames_from_segments(segments, df=None) -> dict[str, pd.DataFrame]:
    """Reconstruye un DataFrame por bloque a partir de los segmentos de Bahamut.

    Con `df` se rehidratan los bloques desde el DataFrame original usando los
    indices del bundle, lo que recupera las columnas que Bahamut dejo fuera de
    `feature_cols`. Sin el, cada bloque se arma con X_* mas y_*.
    """

    if not _looks_like_bahamut_segments(segments):
        raise TypeError(
            "Esto no parece un bundle de Bahamut: falta 'X_train' o los indices por bloque. "
            "Pasa el diccionario que devuelve BahamutSplit.ejecutar_segmentacion()."
        )

    frames: dict[str, pd.DataFrame] = {}

    for split_name, x_key, y_key, index_key in _BAHAMUT_SPLITS:
        x_part = segments.get(x_key)
        if x_part is None:
            continue

        if df is not None:
            if not df.index.is_unique:
                raise ValueError(
                    "df debe tener un indice unico para rehidratar los bloques de Bahamut. "
                    "Usa reset_index(drop=True) antes de crear los segmentos."
                )
            indices = segments.get(index_key)
            if indices is None:
                raise ValueError(
                    f"Para rehidratar el bloque '{split_name}' desde df hace falta '{index_key}' en el bundle."
                )
            frames[split_name] = df.loc[list(indices)].copy()
            continue

        if not isinstance(x_part, pd.DataFrame):
            raise TypeError(f"La clave '{x_key}' de Bahamut deberia ser un pandas.DataFrame.")

        frame = x_part.copy()
        target_frame = _coerce_bahamut_target(segments.get(y_key), split_name)
        if target_frame is not None:
            overlap = sorted(set(frame.columns) & set(target_frame.columns))
            if overlap:
                raise ValueError(
                    f"El bloque '{split_name}' de Bahamut tiene columnas repetidas entre X e y: {overlap}"
                )
            if not frame.index.equals(target_frame.index):
                raise ValueError(f"Los indices de X e y no coinciden en el bloque '{split_name}'.")
            # Asignacion posicional: join multiplica filas con indices repetidos.
            for column in target_frame.columns:
                frame[column] = target_frame[column].to_numpy()
        frames[split_name] = frame

    if "train" not in frames:
        raise ValueError("El bundle de Bahamut no contiene bloque de entrenamiento.")

    return frames


class Fenrir:
    """Pipeline de clustering guiado por PCA con diagnosticos interpretables."""

    def __init__(
        self,
        data,
        target=None,
        features: list[str] | tuple[str, ...] | None = None,
        exclude_columns: list[str] | tuple[str, ...] | set[str] | None = None,
        variance_threshold: float = 0.9,
        cluster_range: range | list[int] | tuple[int, ...] = range(2, 9),
        algorithms: tuple[str, ...] | list[str] = ("kmeans", "gmm", "agglomerative"),
        scalers: dict[str, Any] | None = None,
        max_categories: int = 20,
        random_state: int = 42,
        strict_predict_schema: bool = True,
    ):
        self.data = data.copy() if hasattr(data, "copy") else data
        self.target = target
        self.features = list(features) if features is not None else None
        self.exclude_columns = set(exclude_columns or [])
        self.variance_threshold = float(variance_threshold)
        self.cluster_range = tuple(sorted({int(value) for value in cluster_range}))
        self.algorithms = tuple(algorithms)
        self.scalers = scalers or {
            "StandardScaler": StandardScaler(),
            "RobustScaler": RobustScaler(),
            "MinMaxScaler": MinMaxScaler(),
        }
        self.max_categories = int(max_categories)
        self.random_state = random_state
        # Con True (por defecto) predecir sobre un frame al que le faltan
        # columnas de entrenamiento falla en vez de rellenarlas con la mediana
        # del entrenamiento, que devuelve clusters de aspecto normal calculados
        # sobre datos que el usuario nunca aporto.
        self.strict_predict_schema = bool(strict_predict_schema)

        self.imputer_ = None
        self.feature_names_ = None
        self.feature_source_map_ = {}
        self.target_ = None
        self.target_name_: str = target if isinstance(target, str) else "target"
        self.raw_training_frame_ = None
        self.training_frame_ = None
        self.preprocessing_report_ = pd.DataFrame(columns=["column", "reason", "detail"])
        self.pca_profiles_ = {}
        self.cluster_models_ = {}
        self.cluster_labels_ = {}
        self.search_results_ = pd.DataFrame()
        self.search_failures_ = pd.DataFrame()
        self.scaler_results_ = pd.DataFrame()
        self.best_result_ = None
        self.best_model_ = None
        self.best_labels_ = None

        self.holdout_results_ = pd.DataFrame()
        self.holdout_summary_ = pd.DataFrame()
        self.influential_subset_search_ = pd.DataFrame()
        self.influential_subset_models_ = {}
        self.best_influential_subset_ = None
        self.best_influential_variables_ = []
        self.best_influential_fenrir_ = None
        self.stability_results_ = pd.DataFrame()
        self.stability_summary_ = pd.DataFrame()
        self.analysis_report_ = {}

        # Bundle de Bahamut cuando la instancia nace de from_bahamut().
        self.bahamut_segments_ = None
        self.bahamut_frames_: dict[str, pd.DataFrame] = {}
        self.bahamut_split_results_ = pd.DataFrame()

    @classmethod
    def from_bahamut(
        cls,
        segments,
        *,
        df=None,
        target=None,
        fit: bool = False,
        **kwargs,
    ) -> "Fenrir":
        """Construye un Fenrir sobre el bloque de entrenamiento de un bundle de Bahamut.

        Bahamut decide los cortes; Fenrir los respeta. La instancia se ajusta solo
        con `train`, y `evaluate_bahamut_splits()` puntua validation y test como
        lo que son: bloques que el modelo no ha visto. Es la alternativa honesta a
        `evaluate_holdout()`, que se inventa sus propias particiones aleatorias e
        ignora la segmentacion que ya habias decidido aguas arriba.

        Parameters
        ----------
        segments
            El diccionario que devuelve `BahamutSplit.ejecutar_segmentacion()`.
        df
            DataFrame original, opcional. Con el, cada bloque se rehidrata por
            indices y recupera las columnas que Bahamut dejo fuera de feature_cols.
        target
            Columna objetivo. Si se omite se deduce de `y_train` cuando este trae
            una sola columna.
        fit
            Ajusta antes de devolver la instancia.
        """

        target_name = _resolve_bahamut_target_name(segments, target=target)
        frames = _bahamut_frames_from_segments(segments, df=df)

        train_frame = frames["train"]
        if target_name is not None and target_name not in train_frame.columns:
            raise ValueError(
                f"La columna objetivo '{target_name}' no esta en el bloque de entrenamiento. "
                "Pasa df=dataframe_original si esa columna no viaja dentro del bundle."
            )

        # Rehidratar df no debe reintroducir identificadores, exclusiones u otros
        # objetivos como predictoras. Un features explicito sigue siendo posible.
        if kwargs.get("features") is None:
            kwargs["features"] = list(segments["X_train"].columns)
        model = cls(train_frame, target=target_name, **kwargs)
        model.bahamut_segments_ = segments
        model.bahamut_frames_ = frames

        if fit:
            model.fit()
        return model

    @staticmethod
    def metric_reference():
        return pd.DataFrame(
            [
                {
                    "metric": "explained_variance",
                    "direction": "mas alto es mejor",
                    "acceptable": "0.70+ aceptable | 0.85+ fuerte | 0.90+ muy fuerte",
                    "notes": "No garantiza clusters buenos por si solo; mide cuanta informacion conserva el PCA.",
                },
                {
                    "metric": "variance_first_two",
                    "direction": "mas alto es mejor",
                    "acceptable": "0.50+ suele permitir graficos 2D legibles",
                    "notes": "Sirve sobre todo para visualizacion; no sustituye la calidad del modelo completo.",
                },
                {
                    "metric": "silhouette",
                    "direction": "mas alto es mejor",
                    "acceptable": "<0.25 flojo | 0.25-0.50 usable | 0.50-0.70 bueno | 0.70+ excelente",
                    "notes": "Es la referencia mas interpretable para separacion y compacidad interna.",
                },
                {
                    "metric": "davies_bouldin",
                    "direction": "mas bajo es mejor",
                    "acceptable": "<0.60 excelente | <1.00 bueno | 1.00-1.50 regular | 1.50+ flojo",
                    "notes": "No tiene umbral universal, pero penaliza clusters poco separados.",
                },
                {
                    "metric": "calinski_harabasz",
                    "direction": "mas alto es mejor",
                    "acceptable": "sin umbral absoluto",
                    "notes": "Se interpreta comparando configuraciones dentro del mismo dataset.",
                },
                {
                    "metric": "ari",
                    "direction": "mas alto es mejor",
                    "acceptable": "0.30+ senal | 0.50+ bueno | 0.75+ muy fuerte",
                    "notes": "Mide acuerdo con la variable objetivo externa sin depender del nombre del cluster.",
                },
                {
                    "metric": "nmi",
                    "direction": "mas alto es mejor",
                    "acceptable": "0.30+ senal | 0.50+ bueno | 0.75+ muy fuerte",
                    "notes": "Mide informacion compartida entre clusters y objetivo.",
                },
                {
                    "metric": "v_measure",
                    "direction": "mas alto es mejor",
                    "acceptable": "0.30+ senal | 0.50+ bueno | 0.75+ muy fuerte",
                    "notes": "Equilibra homogeneidad y completitud respecto al objetivo.",
                },
                {
                    "metric": "n_clusters",
                    "direction": "depende del problema",
                    "acceptable": "aceptable si es estable entre splits",
                    "notes": "Muchos clusters pequenos suelen ser fragiles aunque la silhouette suba.",
                },
                {
                    "metric": "n_components",
                    "direction": "menos es mejor si el rendimiento no cae",
                    "acceptable": "mantener solo las necesarias",
                    "notes": "Sirve para balancear compresion, interpretabilidad y estabilidad.",
                },
            ]
        )

    def _as_frame(self, data):
        if isinstance(data, pd.DataFrame):
            return data.copy()
        return pd.DataFrame(data).copy()

    def _extract_target(self, frame, fit=False):
        target_series = None

        if isinstance(self.target, str):
            if self.target in frame.columns:
                target_series = frame.pop(self.target).rename(self.target)
            elif fit:
                raise ValueError(f"La columna objetivo '{self.target}' no existe en los datos.")
        elif self.target is not None:
            target_series = pd.Series(self.target, index=frame.index, name=self.target_name_)

        if target_series is not None:
            self.target_name_ = str(target_series.name or "target")

        return frame, target_series

    def _select_raw_columns(self, frame, fit=False):
        records = []

        if self.features is not None:
            missing = [column for column in self.features if column not in frame.columns]
            if missing and fit:
                raise ValueError(f"Las columnas indicadas en features no existen: {missing}")

            selected = [column for column in self.features if column in frame.columns]
            removed = [column for column in frame.columns if column not in selected]

            if fit:
                records.extend(
                    {"column": column, "reason": "fuera de features", "detail": None}
                    for column in removed
                )

            frame = frame.loc[:, selected]

        removable = [column for column in frame.columns if column in self.exclude_columns]
        if fit:
            records.extend(
                {"column": column, "reason": "exclusion manual", "detail": None}
                for column in removable
            )

        frame = frame.drop(columns=removable, errors="ignore")
        return frame, records

    @overload
    def _encode_frame(self, frame: pd.DataFrame, fit: Literal[True]) -> tuple[pd.DataFrame, dict[str, str], list[dict[str, Any]]]:
        ...

    @overload
    def _encode_frame(self, frame: pd.DataFrame, fit: Literal[False] = False) -> pd.DataFrame:
        ...

    def _encode_frame(
        self,
        frame: pd.DataFrame,
        fit: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, str], list[dict[str, Any]]]:
        numeric = frame.select_dtypes(include=[np.number]).copy()
        categorical = frame.select_dtypes(exclude=[np.number]).copy()

        encoded_parts = []
        source_map: dict[str, str] = {}
        records: list[dict[str, Any]] = []

        if not numeric.empty:
            encoded_parts.append(numeric)
            source_map.update({column: column for column in numeric.columns})

        for column in categorical.columns:
            levels = int(categorical[column].nunique(dropna=True))
            if levels == 0:
                records.append({"column": column, "reason": "sin niveles validos", "detail": 0})
                continue
            if levels > self.max_categories:
                records.append({"column": column, "reason": "alta cardinalidad", "detail": levels})
                continue

            dummies = pd.get_dummies(
                categorical[[column]],
                prefix_sep="__",
                dummy_na=True,
            )
            encoded_parts.append(dummies)
            source_map.update({dummy_column: column for dummy_column in dummies.columns})

        if not encoded_parts:
            raise ValueError("No hay variables utilizables para construir el PCA.")

        prepared = pd.concat(encoded_parts, axis=1)
        if fit:
            return prepared, source_map, records
        return prepared

    def _prepare_frame(self, data, fit=False):
        frame = self._as_frame(data)
        frame, target_series = self._extract_target(frame, fit=fit)
        frame, selected_records = self._select_raw_columns(frame, fit=fit)
        frame = frame.replace([np.inf, -np.inf], np.nan)

        if fit:
            self.raw_training_frame_ = frame.copy()
            prepared, source_map, encoded_records = self._encode_frame(frame, fit=True)
            variable_mask = prepared.nunique(dropna=False) > 1
            prepared = prepared.loc[:, variable_mask]

            if prepared.shape[1] < 2:
                raise ValueError("El PCA necesita al menos dos variables informativas.")

            self.feature_names_ = prepared.columns.tolist()
            self.imputer_ = SimpleImputer(strategy="median")
            prepared = pd.DataFrame(
                self.imputer_.fit_transform(prepared),
                columns=self.feature_names_,
                index=prepared.index,
            )
            self.feature_source_map_ = {column: source_map[column] for column in self.feature_names_}
            self.training_frame_ = prepared
            self.target_ = target_series
            self.preprocessing_report_ = pd.DataFrame(selected_records + encoded_records)
            return prepared

        if self.imputer_ is None or self.feature_names_ is None:
            raise RuntimeError("Fenrir todavia no ha sido ajustado.")

        self._check_predict_schema(frame)
        prepared = self._encode_frame(frame, fit=False)
        prepared = self._align_to_training_schema(prepared)
        transformed_values = np.asarray(self.imputer_.transform(prepared))
        prepared = pd.DataFrame(
            transformed_values,
            columns=self.feature_names_,
            index=prepared.index,
        )
        return prepared

    def _training_source_columns(self) -> list[str]:
        """Columnas originales que alimentaron el ajuste, sin duplicados."""

        return list(dict.fromkeys(self.feature_source_map_.values()))

    def _check_predict_schema(self, frame) -> None:
        """Exige que el frame de prediccion traiga las columnas del entrenamiento."""

        missing = [column for column in self._training_source_columns() if column not in frame.columns]
        if not missing:
            return

        message = (
            "Al frame le faltan columnas usadas en el ajuste: "
            f"{missing}. Sin ellas Fenrir tendria que inventarlas con la mediana "
            "del entrenamiento y los clusters resultantes no describirian tus datos."
        )
        if self.strict_predict_schema:
            raise ValueError(
                message + " Aportalas, o construye el modelo con strict_predict_schema=False "
                "si de verdad quieres imputarlas."
            )

        import warnings

        warnings.warn(message + " Se imputan por mediana (strict_predict_schema=False).", UserWarning, stacklevel=3)

    def _align_to_training_schema(self, prepared):
        """Reindexa al esquema de entrenamiento distinguiendo dummies de numericas.

        Un nivel categorico ausente en los datos nuevos significa "esa categoria
        no aparece" y su dummy vale 0. Rellenarlo con NaN hacia que el imputador
        le asignase la mediana de la dummy (p. ej. 0.3), inventando una pertenencia
        parcial a una categoria inexistente.
        """

        aligned = prepared.reindex(columns=self.feature_names_)
        for column in self.feature_names_:
            if column in prepared.columns:
                continue
            is_dummy = self.feature_source_map_.get(column, column) != column
            aligned[column] = 0.0 if is_dummy else np.nan
        return aligned.loc[:, self.feature_names_]

    def _build_pca_profile(self, scaler_name, scaler, prepared):
        scaled_values = scaler.fit_transform(prepared)
        scaled_frame = pd.DataFrame(
            scaled_values,
            columns=self.feature_names_,
            index=prepared.index,
        )

        pca = PCA()
        scores = pca.fit_transform(scaled_frame)
        component_names = [f"PC{i}" for i in range(1, scores.shape[1] + 1)]
        explained = pca.explained_variance_ratio_
        cumulative = explained.cumsum()
        selected_components = int(np.searchsorted(cumulative, self.variance_threshold) + 1)
        selected_components = max(2, min(selected_components, scaled_frame.shape[1]))

        scores_df = pd.DataFrame(scores, columns=component_names, index=prepared.index)
        reduced_scores = scores_df.iloc[:, :selected_components].copy()

        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loadings_df = pd.DataFrame(loadings, index=self.feature_names_, columns=component_names)
        contributions = loadings_df.pow(2).divide(loadings_df.pow(2).sum(axis=0), axis=1) * 100
        communality = loadings_df.pow(2).sum(axis=1).replace(0, np.nan)
        cos2 = loadings_df.pow(2).divide(communality, axis=0).fillna(0.0)

        component_report = pd.DataFrame(
            {
                "component": np.arange(1, len(component_names) + 1),
                "explained_variance": explained,
                "cumulative_variance": cumulative,
            }
        )

        return {
            "scaler_name": scaler_name,
            "scaler": scaler,
            "scaled_frame": scaled_frame,
            "pca": pca,
            "scores": scores_df,
            "reduced_scores": reduced_scores,
            "component_names": component_names,
            "loadings": loadings_df,
            "contributions": contributions,
            "cos2": cos2,
            "component_report": component_report,
            "selected_components": selected_components,
            "selected_variance": float(cumulative[selected_components - 1]),
            "variance_first_two": float(cumulative[min(1, len(cumulative) - 1)]),
        }

    def _project_with_profile(self, prepared, profile):
        scaled = pd.DataFrame(
            profile["scaler"].transform(prepared),
            columns=self.feature_names_,
            index=prepared.index,
        )
        projected = profile["pca"].transform(scaled)
        return pd.DataFrame(projected, columns=profile["component_names"], index=prepared.index)

    def _resolve_components(self, profile, components=None):
        if components is None:
            component_numbers = tuple(range(1, int(profile["selected_components"]) + 1))
        elif isinstance(components, int):
            component_numbers = (int(components),)
        else:
            component_numbers = tuple(int(value) for value in components)

        component_names = [f"PC{value}" for value in component_numbers]
        missing = [name for name in component_names if name not in profile["loadings"].columns]
        if missing:
            raise ValueError(f"Componentes no disponibles: {missing}")
        return component_numbers, component_names

    def _feature_source_series(self):
        return pd.Series(self.feature_source_map_)

    def _fit_candidate(self, algorithm, n_clusters, reduced_scores):
        if algorithm == "kmeans":
            model = KMeans(n_clusters=n_clusters, n_init=20, random_state=self.random_state)
            labels = model.fit_predict(reduced_scores)
        elif algorithm == "gmm":
            model = GaussianMixture(
                n_components=n_clusters,
                covariance_type="full",
                n_init=5,
                random_state=self.random_state,
            )
            labels = model.fit_predict(reduced_scores)
        elif algorithm == "agglomerative":
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
            labels = model.fit_predict(reduced_scores)
        else:
            raise ValueError(f"Algoritmo no soportado: {algorithm}")

        unique_labels = np.unique(labels)
        if unique_labels.size < 2 or unique_labels.size >= len(reduced_scores):
            raise ValueError("Configuracion invalida para calcular silhouette.")

        silhouette = silhouette_score(
            reduced_scores,
            labels,
            sample_size=min(2000, len(reduced_scores)),
            random_state=self.random_state,
        )
        calinski = calinski_harabasz_score(reduced_scores, labels)
        davies = davies_bouldin_score(reduced_scores, labels)
        return model, labels, silhouette, calinski, davies

    def _require_fit(self):
        if self.best_result_ is None:
            raise RuntimeError("Ejecuta fit() antes de consultar resultados.")

    def _require_best_result(self) -> dict[str, Any]:
        self._require_fit()
        if self.best_result_ is None:
            raise RuntimeError("No hay configuracion base disponible.")
        return {str(key): value for key, value in dict(self.best_result_).items()}

    def _require_best_labels(self) -> pd.Series:
        self._require_fit()
        if self.best_labels_ is None:
            raise RuntimeError("No hay etiquetas de clusters disponibles.")
        return self.best_labels_

    def _require_best_model(self):
        self._require_fit()
        if self.best_model_ is None:
            raise RuntimeError("No hay modelo base disponible.")
        return self.best_model_

    def _require_training_frame(self) -> pd.DataFrame:
        self._require_fit()
        if self.training_frame_ is None:
            raise RuntimeError("No hay frame de entrenamiento disponible.")
        return self.training_frame_

    def _require_raw_training_frame(self) -> pd.DataFrame:
        self._require_fit()
        if self.raw_training_frame_ is None:
            raise RuntimeError("No hay frame bruto de entrenamiento disponible.")
        return self.raw_training_frame_

    def _require_target_series(self) -> pd.Series:
        self._require_fit()
        if self.target_ is None:
            raise RuntimeError("No hay target disponible en este modelo.")
        return self.target_

    def _result_for_scaler(self, scaler_name=None):
        best_result = self._require_best_result()
        if scaler_name is None:
            return best_result

        subset = self.search_results_[self.search_results_["scaler_name"] == scaler_name]
        if subset.empty:
            raise ValueError(f"No hay resultados para el escalado '{scaler_name}'.")
        return subset.iloc[0].to_dict()

    def _profile_for(self, scaler_name=None):
        result = self._result_for_scaler(scaler_name=scaler_name)
        return self.pca_profiles_[str(result["scaler_name"])]

    def _score_internal(self, reduced_scores, labels, prefix=""):
        result = {
            f"{prefix}silhouette": np.nan,
            f"{prefix}calinski_harabasz": np.nan,
            f"{prefix}davies_bouldin": np.nan,
        }

        unique_labels = np.unique(labels)
        if unique_labels.size < 2 or unique_labels.size >= len(reduced_scores):
            return result

        result[f"{prefix}silhouette"] = float(
            silhouette_score(
                reduced_scores,
                labels,
                sample_size=min(2000, len(reduced_scores)),
                random_state=self.random_state,
            )
        )
        result[f"{prefix}calinski_harabasz"] = float(calinski_harabasz_score(reduced_scores, labels))
        result[f"{prefix}davies_bouldin"] = float(davies_bouldin_score(reduced_scores, labels))
        return result

    def _score_external(self, target_series, labels, index=None, prefix=""):
        result = {
            f"{prefix}ari": np.nan,
            f"{prefix}nmi": np.nan,
            f"{prefix}v_measure": np.nan,
        }

        if target_series is None:
            return result

        target = pd.Series(target_series, index=index if index is not None else getattr(target_series, "index", None))
        labels_series = pd.Series(labels, index=target.index)
        mask = target.notna()
        if mask.sum() < 2:
            return result
        if target[mask].nunique() < 2 or labels_series[mask].nunique() < 2:
            return result

        result[f"{prefix}ari"] = float(adjusted_rand_score(target[mask], labels_series[mask]))
        result[f"{prefix}nmi"] = float(normalized_mutual_info_score(target[mask], labels_series[mask]))
        result[f"{prefix}v_measure"] = float(v_measure_score(target[mask], labels_series[mask]))
        return result

    def _assign_candidate_labels(self, model, algorithm, train_projection, train_labels, test_projection):
        if algorithm in {"kmeans", "gmm"} and hasattr(model, "predict"):
            return model.predict(test_projection)

        train_with_labels = train_projection.copy()
        train_with_labels["cluster"] = np.asarray(train_labels)
        centroids = train_with_labels.groupby("cluster").mean()
        distances = ((test_projection.to_numpy()[:, None, :] - centroids.to_numpy()[None, :, :]) ** 2).sum(axis=2)
        return centroids.index.to_numpy()[distances.argmin(axis=1)]

    def _new_like(self, data, target_name=None):
        return Fenrir(
            data,
            target=self.target if isinstance(self.target, str) else target_name,
            features=self.features,
            exclude_columns=sorted(self.exclude_columns),
            variance_threshold=self.variance_threshold,
            cluster_range=self.cluster_range,
            algorithms=self.algorithms,
            scalers={name: clone(scaler) for name, scaler in self.scalers.items()},
            max_categories=self.max_categories,
            random_state=self.random_state,
            strict_predict_schema=self.strict_predict_schema,
        )

    def fit(self):
        if not 0 < self.variance_threshold <= 1:
            raise ValueError("variance_threshold debe estar en el intervalo (0, 1].")

        valid_algorithms = {"kmeans", "gmm", "agglomerative"}
        unknown = set(self.algorithms) - valid_algorithms
        if unknown:
            raise ValueError(f"Algoritmos no soportados: {sorted(unknown)}")

        self.pca_profiles_ = {}
        self.cluster_models_ = {}
        self.cluster_labels_ = {}
        self.search_results_ = pd.DataFrame()
        self.scaler_results_ = pd.DataFrame()
        self.best_result_ = None
        self.best_model_ = None
        self.best_labels_ = None

        prepared = self._prepare_frame(self.data, fit=True)
        max_valid_clusters = len(prepared) - 1
        cluster_values = [value for value in self.cluster_range if 2 <= value <= max_valid_clusters]
        if not cluster_values:
            raise ValueError("cluster_range debe contener valores entre 2 y n_samples - 1.")

        candidate_rows = []
        scaler_rows = []
        failure_rows = []

        for scaler_name, base_scaler in self.scalers.items():
            scaler = clone(base_scaler)
            profile = self._build_pca_profile(scaler_name, scaler, prepared)
            self.pca_profiles_[scaler_name] = profile
            scaler_candidates = []

            for algorithm in self.algorithms:
                for n_clusters in cluster_values:
                    try:
                        model, labels, silhouette, calinski, davies = self._fit_candidate(
                            algorithm,
                            n_clusters,
                            profile["reduced_scores"],
                        )
                    except Exception as exc:
                        # Un candidato invalido (p. ej. k mayor que el numero de
                        # puntos distintos) no debe tumbar la busqueda, pero
                        # tampoco puede desaparecer sin dejar rastro.
                        failure_rows.append(
                            {
                                "scaler_name": scaler_name,
                                "algorithm": algorithm,
                                "n_clusters": int(n_clusters),
                                "n_components": int(profile["selected_components"]),
                                "error": type(exc).__name__,
                                "detail": str(exc),
                            }
                        )
                        continue

                    key = (
                        scaler_name,
                        algorithm,
                        int(n_clusters),
                        int(profile["selected_components"]),
                    )
                    self.cluster_models_[key] = model
                    self.cluster_labels_[key] = labels

                    row = {
                        "scaler_name": scaler_name,
                        "algorithm": algorithm,
                        "n_clusters": int(n_clusters),
                        "n_components": int(profile["selected_components"]),
                        "explained_variance": float(profile["selected_variance"]),
                        "variance_first_two": float(profile["variance_first_two"]),
                        "silhouette": float(silhouette),
                        "calinski_harabasz": float(calinski),
                        "davies_bouldin": float(davies),
                    }
                    candidate_rows.append(row)
                    scaler_candidates.append(row)

            if scaler_candidates:
                scaler_df = pd.DataFrame(scaler_candidates).sort_values(
                    ["silhouette", "explained_variance", "calinski_harabasz", "davies_bouldin", "n_components"],
                    ascending=[False, False, False, True, True],
                )
                scaler_rows.append(scaler_df.iloc[0].to_dict())

        self.search_failures_ = pd.DataFrame(failure_rows)

        if not candidate_rows:
            motivos = (
                "; ".join(sorted({f"{row['algorithm']}/k={row['n_clusters']}: {row['detail']}" for row in failure_rows}))
                or "no se genero ningun candidato"
            )
            raise RuntimeError(f"No se pudo ajustar ningun modelo de clustering valido. Motivos: {motivos}")

        self.search_results_ = pd.DataFrame(candidate_rows).sort_values(
            ["silhouette", "explained_variance", "calinski_harabasz", "davies_bouldin", "n_components"],
            ascending=[False, False, False, True, True],
        ).reset_index(drop=True)
        self.scaler_results_ = pd.DataFrame(scaler_rows).sort_values(
            ["silhouette", "explained_variance", "calinski_harabasz", "davies_bouldin", "n_components"],
            ascending=[False, False, False, True, True],
        ).reset_index(drop=True)

        self.best_result_ = self.search_results_.iloc[0].to_dict()
        best_key = (
            self.best_result_["scaler_name"],
            self.best_result_["algorithm"],
            int(self.best_result_["n_clusters"]),
            int(self.best_result_["n_components"]),
        )
        self.best_model_ = self.cluster_models_[best_key]
        self.best_labels_ = pd.Series(
            self.cluster_labels_[best_key],
            index=prepared.index,
            name="cluster",
        )
        return self

    def fit_predict(self):
        self.fit()
        return self._require_best_labels().copy()

    def predict(self, data=None):
        best_result = self._require_best_result()
        best_labels = self._require_best_labels()
        if data is None:
            return best_labels.copy()

        if best_result["algorithm"] == "agglomerative":
            raise ValueError(
                "El mejor modelo es agglomerative y no admite prediccion fuera de muestra. "
                "Restringe algorithms a ('kmeans', 'gmm') si necesitas predict sobre nuevos datos."
            )

        prepared = self._prepare_frame(data, fit=False)
        profile = self._profile_for()
        transformed = self._project_with_profile(prepared, profile).iloc[:, : int(best_result["n_components"])]
        labels = self._require_best_model().predict(transformed)
        return pd.Series(labels, index=prepared.index, name="cluster")

    def transform(self, data=None, scaler_name=None, n_components=None):
        self._require_fit()
        profile = self._profile_for(scaler_name=scaler_name)

        if data is None:
            prepared = self._require_training_frame()
        else:
            prepared = self._prepare_frame(data, fit=False)

        transformed = self._project_with_profile(prepared, profile)
        if n_components is None:
            n_components = int(profile["selected_components"])
        return transformed.iloc[:, : int(n_components)].copy()

    def search_failures_report(self):
        """Candidatos descartados durante `fit()` y por que."""

        self._require_fit()
        if self.search_failures_.empty:
            return pd.DataFrame(
                [{"scaler_name": "(ninguno)", "algorithm": None, "n_clusters": None, "error": None, "detail": "todos los candidatos se ajustaron"}]
            )
        return self.search_failures_.copy()

    def preprocessing_report(self):
        self._require_fit()
        if self.preprocessing_report_.empty:
            return pd.DataFrame([{"column": "(ninguna)", "reason": "sin descartes", "detail": None}])
        return self.preprocessing_report_.copy()

    def best_configuration(self):
        return pd.Series(self._require_best_result(), name="mejor_configuracion")

    def summary(self, top_n=10):
        self._require_fit()
        columns = [
            "scaler_name",
            "algorithm",
            "n_clusters",
            "n_components",
            "explained_variance",
            "variance_first_two",
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
        ]
        frame = self.search_results_.loc[:, columns].head(top_n).copy()
        return _decorate_candidate_table(frame)

    def scaler_report(self):
        self._require_fit()
        columns = [
            "scaler_name",
            "algorithm",
            "n_clusters",
            "n_components",
            "explained_variance",
            "variance_first_two",
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
        ]
        frame = self.scaler_results_.loc[:, columns].copy()
        return _decorate_candidate_table(frame)

    def pca_report(self, scaler_name=None):
        profile = self._profile_for(scaler_name=scaler_name)
        return profile["component_report"].copy()

    def feature_report(self, scaler_name=None, components=None, top_n: int | None = 10):
        profile = self._profile_for(scaler_name=scaler_name)
        component_numbers, component_names = self._resolve_components(profile, components)

        loadings = profile["loadings"].loc[:, component_names].copy()
        contributions = profile["contributions"].loc[:, component_names].copy()
        cos2 = profile["cos2"].loc[:, component_names].copy()

        weights = pd.Series(
            profile["component_report"].set_index("component").loc[list(component_numbers), "explained_variance"].values,
            index=component_names,
        )

        report = pd.concat(
            [
                loadings.rename(columns=lambda column: f"{column}_loading"),
                contributions.rename(columns=lambda column: f"{column}_contribution"),
                cos2.rename(columns=lambda column: f"{column}_cos2"),
            ],
            axis=1,
        )
        report["contribution_total"] = contributions.sum(axis=1)
        report["cos2_total"] = cos2.sum(axis=1)
        report["weighted_contribution"] = contributions.mul(weights, axis=1).sum(axis=1) / weights.sum()
        report["weighted_cos2"] = cos2.mul(weights, axis=1).sum(axis=1) / weights.sum()
        report = report.sort_values(["weighted_contribution", "weighted_cos2"], ascending=[False, False])
        if top_n is not None:
            return report.head(top_n)
        return report

    def source_feature_report(self, scaler_name=None, components=None, top_n: int | None = 10):
        profile = self._profile_for(scaler_name=scaler_name)
        component_numbers, component_names = self._resolve_components(profile, components)
        source_series = self._feature_source_series()

        grouped_loadings = profile["loadings"].loc[:, component_names].abs().groupby(source_series).max()
        grouped_contributions = profile["contributions"].loc[:, component_names].groupby(source_series).sum()
        grouped_cos2 = profile["cos2"].loc[:, component_names].groupby(source_series).max()
        encoded_counts = source_series.groupby(source_series).size().rename("encoded_features")

        weights = pd.Series(
            profile["component_report"].set_index("component").loc[list(component_numbers), "explained_variance"].values,
            index=component_names,
        )

        report = pd.concat(
            [
                grouped_loadings.rename(columns=lambda column: f"{column}_loading_absmax"),
                grouped_contributions.rename(columns=lambda column: f"{column}_contribution"),
                grouped_cos2.rename(columns=lambda column: f"{column}_cos2_max"),
                encoded_counts,
            ],
            axis=1,
        )
        report["contribution_total"] = grouped_contributions.sum(axis=1)
        report["cos2_total"] = grouped_cos2.sum(axis=1)
        report["weighted_contribution"] = grouped_contributions.mul(weights, axis=1).sum(axis=1) / weights.sum()
        report["weighted_cos2"] = grouped_cos2.mul(weights, axis=1).sum(axis=1) / weights.sum()
        report = report.sort_values(["weighted_contribution", "weighted_cos2"], ascending=[False, False])
        if top_n is not None:
            return report.head(top_n)
        return report

    @overload
    def influential_variables(
        self,
        scaler_name=None,
        components=None,
        top_n: int | None = 10,
        grouped: bool = True,
        *,
        as_list: Literal[True],
    ) -> list[str]:
        ...

    @overload
    def influential_variables(
        self,
        scaler_name=None,
        components=None,
        top_n: int | None = 10,
        grouped: bool = True,
        as_list: Literal[False] = False,
    ) -> pd.DataFrame:
        ...

    def influential_variables(
        self,
        scaler_name=None,
        components=None,
        top_n: int | None = 10,
        grouped=True,
        as_list=False,
    ) -> pd.DataFrame | list[str]:
        if grouped:
            report = self.source_feature_report(
                scaler_name=scaler_name,
                components=components,
                top_n=top_n,
            )
        else:
            report = self.feature_report(
                scaler_name=scaler_name,
                components=components,
                top_n=top_n,
            )

        if as_list:
            return report.index.tolist()
        return report

    def list_influential_variables(self, scaler_name=None, components=None, top_n: int | None = 10, grouped=True):
        return self.influential_variables(
            scaler_name=scaler_name,
            components=components,
            top_n=top_n,
            grouped=grouped,
            as_list=True,
        )

    def influential_dataset(self, data=None, top_n=10, components=None, include_target=False):
        self._require_fit()

        if data is None:
            frame = self._require_raw_training_frame().copy()
            target_series = None if self.target_ is None else self.target_.copy()
        else:
            frame = self._as_frame(data)
            frame, target_series = self._extract_target(frame, fit=False)
            frame, _ = self._select_raw_columns(frame, fit=False)

        selected = [
            column
            for column in self.influential_variables(
                components=components,
                top_n=top_n,
                grouped=True,
                as_list=True,
            )
            if column in frame.columns
        ]

        if not selected:
            raise ValueError("No se pudieron recuperar variables originales a partir de la influencia del PCA.")

        subset = frame.loc[:, selected].copy()
        if include_target and target_series is not None:
            subset[self.target_name_] = target_series.values
        return subset

    def refit_with_influential_variables(
        self,
        data=None,
        top_n=10,
        components=None,
        fit=True,
        **overrides,
    ):
        target_name = overrides.pop(
            "target",
            self.target_name_ if self.target_ is not None else None,
        )
        subset = self.influential_dataset(
            data=data,
            top_n=top_n,
            components=components,
            include_target=target_name is not None,
        )

        next_fenrir = Fenrir(
            subset,
            target=target_name,
            features=overrides.pop("features", None),
            exclude_columns=overrides.pop("exclude_columns", None),
            variance_threshold=overrides.pop("variance_threshold", self.variance_threshold),
            cluster_range=overrides.pop("cluster_range", self.cluster_range),
            algorithms=overrides.pop("algorithms", self.algorithms),
            scalers=overrides.pop(
                "scalers",
                {name: clone(scaler) for name, scaler in self.scalers.items()},
            ),
            max_categories=overrides.pop("max_categories", self.max_categories),
            random_state=overrides.pop("random_state", self.random_state),
        )

        if overrides:
            raise TypeError(f"Argumentos no utilizados: {sorted(overrides)}")

        if fit:
            next_fenrir.fit()
        return next_fenrir

    def cluster_target_report(self, normalize: bool | Literal["all", "index", "columns"] = "index"):
        self._require_fit()
        if self.target_ is None:
            raise ValueError("No has indicado una variable objetivo para perfilar los clusters.")
        best_labels = self._require_best_labels()
        target_series = self._require_target_series()

        base = pd.crosstab(
            best_labels.rename("cluster"),
            target_series.rename(str(self.target_name_)),
        )
        normalized = base.div(base.sum(axis=1).replace(0, np.nan), axis=0)
        report = pd.crosstab(
            best_labels.rename("cluster"),
            target_series.rename(str(self.target_name_)),
            normalize=normalize,
        )
        report["target_dominante"] = normalized.idxmax(axis=1)
        report["share_dominante"] = normalized.max(axis=1).apply(lambda value: _safe_round(value, 4))
        report["lectura_target"] = normalized.max(axis=1).map(_interpret_target_concentration)
        return report

    def cluster_frame(self):
        self._require_fit()
        profile = self._profile_for()
        frame = profile["scores"].iloc[:, :2].copy()
        frame["cluster"] = self._require_best_labels().values
        if self.target_ is not None:
            frame[self.target_name_] = self.target_.values
        return frame

    def attach_clusters(
        self,
        data=None,
        column_name="cluster",
        include_components=False,
        n_components=2,
        component_prefix="pca_",
    ):
        self._require_fit()

        if data is None:
            frame = self._as_frame(self.data)
            labels = self._require_best_labels().copy()
            components = self.transform(n_components=n_components) if include_components else None
        else:
            frame = self._as_frame(data)
            labels = self.predict(frame)
            components = self.transform(data=frame, n_components=n_components) if include_components else None

        result = frame.copy()
        result[column_name] = labels.values
        if components is not None:
            result = pd.concat([result, components.add_prefix(component_prefix)], axis=1)
        return result

    def clustered_dataframe(
        self,
        data=None,
        cluster_column="cluster",
        include_target=True,
        include_components=False,
        n_components=2,
        component_prefix="pca_",
    ):
        result = self.attach_clusters(
            data=data,
            column_name=cluster_column,
            include_components=include_components,
            n_components=n_components,
            component_prefix=component_prefix,
        )

        if include_target and self.target_ is not None and self.target_name_ not in result.columns:
            if data is None:
                result[self.target_name_] = self.target_.values
            else:
                candidate = self._as_frame(data)
                if isinstance(self.target, str) and self.target in candidate.columns:
                    result[self.target_name_] = candidate[self.target].values
        return result

    def clustered_frame(self, *args, **kwargs):
        return self.clustered_dataframe(*args, **kwargs)

    def cluster_size_report(self, data=None, cluster_column="cluster", normalize=False):
        frame = self.clustered_dataframe(data=data, cluster_column=cluster_column, include_target=False)
        counts = frame[cluster_column].value_counts(normalize=False).sort_index()
        shares = frame[cluster_column].value_counts(normalize=True).sort_index()

        if normalize:
            report = shares.rename("cluster_share").to_frame()
            report["cluster_size"] = counts
        else:
            report = counts.rename("cluster_size").to_frame()
            report["cluster_share"] = shares

        return _decorate_cluster_size_table(report)

    def cluster_groupby(
        self,
        data=None,
        variables=None,
        top_n=5,
        include_target=True,
        include_size=True,
        numeric_agg="mean",
        categorical_agg="mode",
        cluster_column="cluster",
        round_digits=4,
    ):
        frame = self.clustered_dataframe(
            data=data,
            cluster_column=cluster_column,
            include_target=include_target,
            include_components=False,
        )

        if variables is None:
            variables = self.list_influential_variables(top_n=top_n, grouped=True)
        else:
            variables = list(dict.fromkeys(variables))

        if include_target and self.target_ is not None and self.target_name_ not in variables:
            variables = variables + [self.target_name_]

        available = [column for column in variables if column in frame.columns and column != cluster_column]
        if not available:
            raise ValueError("No hay variables disponibles para resumir por cluster.")

        agg_map = {}
        for column in available:
            if pd.api.types.is_numeric_dtype(frame[column]):
                agg_map[column] = numeric_agg
            elif categorical_agg == "mode":
                agg_map[column] = _mode_or_nan
            else:
                agg_map[column] = categorical_agg

        grouped = frame.groupby(cluster_column).agg(agg_map)
        if include_size:
            sizes = frame.groupby(cluster_column).size()
            grouped.insert(0, "cluster_size", sizes)
            grouped.insert(1, "cluster_share", (sizes / len(frame)).round(round_digits))

        numeric_columns = grouped.select_dtypes(include=[np.number]).columns
        grouped.loc[:, numeric_columns] = grouped.loc[:, numeric_columns].round(round_digits)
        return _decorate_cluster_groupby_table(grouped)

    def metric_report(self, result=None):
        if result is None:
            best_result = self._require_best_result()
            best_labels = self._require_best_labels()
            result_dict = best_result.copy()
            result_dict.update(self._score_external(self.target_, best_labels, index=best_labels.index))
        elif isinstance(result, pd.Series):
            result_dict = result.to_dict()
        else:
            result_dict = dict(result)

        rows = []
        for reference in self.metric_reference().to_dict("records"):
            metric = reference["metric"]
            if metric not in result_dict:
                continue
            rows.append(
                {
                    "metric": metric,
                    "value": result_dict[metric],
                    "direction": reference["direction"],
                    "acceptable": reference["acceptable"],
                    "interpretation": _interpret_metric(metric, result_dict[metric]),
                    "notes": reference["notes"],
                }
            )
        return _decorate_metric_table(pd.DataFrame(rows))

    def _bahamut_frames(self, segments=None, df=None) -> dict[str, pd.DataFrame]:
        if segments is not None:
            return _bahamut_frames_from_segments(segments, df=df)
        if self.bahamut_frames_:
            return self.bahamut_frames_
        if self.bahamut_segments_ is not None:
            return _bahamut_frames_from_segments(self.bahamut_segments_, df=df)
        raise RuntimeError(
            "Esta instancia no nacio de un bundle de Bahamut. Construyela con "
            "Fenrir.from_bahamut(segmentos) o pasa segments= a este metodo."
        )

    def evaluate_bahamut_splits(self, segments=None, df=None, include_train: bool = True):
        """Puntua los bloques de Bahamut con el modelo ajustado solo sobre train.

        A diferencia de `evaluate_holdout()`, aqui no se reajusta nada ni se
        inventan particiones: se proyectan validation y test con el PCA y el
        escalado del ajuste y se les asignan clusters con el modelo base. Es la
        lectura fuera de muestra que corresponde a los cortes que decidio Bahamut.
        """

        self._require_fit()
        frames = self._bahamut_frames(segments=segments, df=df)

        best_result = self._require_best_result()
        best_model = self._require_best_model()
        algorithm = str(best_result["algorithm"])
        n_components = int(best_result["n_components"])

        profile = self._profile_for()
        train_projection = profile["reduced_scores"].iloc[:, :n_components]
        train_labels = np.asarray(self._require_best_labels())

        rows = []
        for split_name in ("train", "validation", "test"):
            frame = frames.get(split_name)
            if frame is None or frame.empty:
                continue
            if split_name == "train" and not include_train:
                continue

            _, target_series = self._extract_target(self._as_frame(frame), fit=False)

            if split_name == "train":
                projection = train_projection
                labels = train_labels
            else:
                prepared = self._prepare_frame(frame, fit=False)
                projection = self._project_with_profile(prepared, profile).iloc[:, :n_components]
                labels = np.asarray(
                    self._assign_candidate_labels(
                        best_model,
                        algorithm,
                        train_projection,
                        train_labels,
                        projection,
                    )
                )

            label_series = pd.Series(labels, index=projection.index, name="cluster")
            shares = label_series.value_counts(normalize=True)

            row = {
                "split": split_name,
                "origen": "ajuste" if split_name == "train" else "fuera de muestra",
                "n_filas": int(len(projection)),
                "n_clusters_observados": int(label_series.nunique()),
                "cluster_min_share": float(shares.min()) if len(shares) else np.nan,
            }
            row.update(self._score_internal(projection, labels))
            row.update(self._score_external(target_series, labels, index=projection.index))
            rows.append(row)

        if not rows:
            raise RuntimeError("El bundle de Bahamut no dejo ningun bloque evaluable.")

        results = pd.DataFrame(rows)
        self.bahamut_split_results_ = results
        return results.copy()

    def bahamut_split_report(self, segments=None, df=None, include_train: bool = True):
        """Version interpretada de `evaluate_bahamut_splits()`."""

        results = (
            self.bahamut_split_results_.copy()
            if segments is None and df is None and not self.bahamut_split_results_.empty
            else self.evaluate_bahamut_splits(segments=segments, df=df, include_train=include_train)
        )
        if not include_train:
            results = results[results["split"] != "train"].reset_index(drop=True)

        table = results.copy()
        table["silhouette_reading"] = table["silhouette"].map(
            lambda value: _interpret_metric("silhouette", value)
        )
        table["davies_bouldin_reading"] = table["davies_bouldin"].map(
            lambda value: _interpret_metric("davies_bouldin", value)
        )
        if "v_measure" in table.columns:
            table["target_alignment_reading"] = table["v_measure"].map(
                lambda value: _interpret_metric("v_measure", value)
            )
        table["cluster_balance"] = table["cluster_min_share"].map(_interpret_cluster_share)

        table = _round_existing_columns(
            table,
            [
                "cluster_min_share",
                "silhouette",
                "calinski_harabasz",
                "davies_bouldin",
                "ari",
                "nmi",
                "v_measure",
            ],
        )

        preferred = [
            "split",
            "origen",
            "n_filas",
            "n_clusters_observados",
            "cluster_min_share",
            "cluster_balance",
            "silhouette",
            "silhouette_reading",
            "davies_bouldin",
            "davies_bouldin_reading",
            "calinski_harabasz",
            "v_measure",
            "target_alignment_reading",
            "nmi",
            "ari",
        ]
        ordered = [column for column in preferred if column in table.columns]
        ordered += [column for column in table.columns if column not in ordered]
        return table.loc[:, ordered]

    def evaluate_holdout(
        self,
        data=None,
        test_size=0.25,
        n_splits=3,
        stratify_target=True,
    ):
        raw = self._as_frame(self.data if data is None else data)
        target_series = None
        target_name = self.target if isinstance(self.target, str) else None

        if isinstance(self.target, str) and self.target in raw.columns:
            target_series = raw[self.target].copy()
        elif self.target is not None:
            target_series = pd.Series(self.target, index=raw.index, name=self.target_name_)
            target_name = self.target_name_
            raw = raw.copy()
            raw[target_name] = target_series.values

        if not 0 < float(test_size) < 1:
            raise ValueError("test_size debe estar entre 0 y 1.")
        if int(n_splits) < 1:
            raise ValueError("n_splits debe ser al menos 1.")

        if stratify_target and target_series is not None and target_series.nunique(dropna=False) > 1:
            stratify_values = target_series.astype(str).where(target_series.notna(), "__missing__")
            splitter = StratifiedShuffleSplit(
                n_splits=int(n_splits),
                test_size=float(test_size),
                random_state=self.random_state,
            )
            iterator = splitter.split(raw, stratify_values)
        else:
            splitter = ShuffleSplit(
                n_splits=int(n_splits),
                test_size=float(test_size),
                random_state=self.random_state,
            )
            iterator = splitter.split(raw)

        results = []
        metric_names = ["silhouette", "calinski_harabasz", "davies_bouldin", "ari", "nmi", "v_measure"]

        for split_id, (train_idx, test_idx) in enumerate(iterator, start=1):
            train_raw = raw.iloc[train_idx].copy()
            test_raw = raw.iloc[test_idx].copy()

            evaluator = self._new_like(train_raw, target_name=target_name)
            evaluator.fit()

            test_prepared = evaluator._prepare_frame(test_raw, fit=False)
            _, test_target = evaluator._extract_target(test_raw.copy(), fit=False)

            for candidate in evaluator.search_results_.to_dict("records"):
                scaler_name = candidate["scaler_name"]
                algorithm = candidate["algorithm"]
                n_clusters = int(candidate["n_clusters"])
                n_components = int(candidate["n_components"])
                key = (scaler_name, algorithm, n_clusters, n_components)

                profile = evaluator.pca_profiles_[scaler_name]
                train_projection = profile["reduced_scores"].iloc[:, :n_components].copy()
                test_projection = evaluator._project_with_profile(test_prepared, profile).iloc[:, :n_components].copy()
                train_labels = evaluator.cluster_labels_[key]
                model = evaluator.cluster_models_[key]
                test_labels = evaluator._assign_candidate_labels(
                    model,
                    algorithm,
                    train_projection,
                    train_labels,
                    test_projection,
                )

                row = candidate.copy()
                row["split"] = split_id
                row["train_size"] = len(train_raw)
                row["test_size"] = len(test_raw)

                for metric_name in metric_names:
                    row[f"train_{metric_name}"] = row.get(metric_name, np.nan)

                row.update(evaluator._score_external(evaluator.target_, train_labels, index=train_projection.index, prefix="train_"))
                row.update(evaluator._score_internal(test_projection, test_labels, prefix="test_"))
                row.update(evaluator._score_external(test_target, test_labels, index=test_projection.index, prefix="test_"))

                for metric_name in metric_names:
                    test_key = f"test_{metric_name}"
                    train_key = f"train_{metric_name}"
                    row[metric_name] = row[test_key] if pd.notna(row[test_key]) else row[train_key]

                results.append(row)

        if not results:
            raise RuntimeError("No se pudo calcular la evaluacion holdout.")

        results_df = pd.DataFrame(results)
        sort_columns = [
            column
            for column in ["silhouette", "explained_variance", "v_measure", "nmi", "calinski_harabasz", "davies_bouldin"]
            if column in results_df.columns
        ]
        ascending = [False, False, False, False, False, True][: len(sort_columns)]
        self.holdout_results_ = results_df.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)

        group_keys = ["scaler_name", "algorithm", "n_clusters", "n_components"]
        aggregate_columns = [
            column
            for column in [
                "explained_variance",
                "variance_first_two",
                "train_silhouette",
                "test_silhouette",
                "train_calinski_harabasz",
                "test_calinski_harabasz",
                "train_davies_bouldin",
                "test_davies_bouldin",
                "train_ari",
                "test_ari",
                "train_nmi",
                "test_nmi",
                "train_v_measure",
                "test_v_measure",
            ]
            if column in results_df.columns
        ]

        summary = results_df.groupby(group_keys, as_index=False)[aggregate_columns].agg(["mean", "std"])
        flat_columns = summary.columns.to_flat_index() if isinstance(summary.columns, pd.MultiIndex) else summary.columns
        summary.columns = [
            column if isinstance(column, str) else f"{column[0]}_{column[1]}".strip("_")
            for column in flat_columns
        ]
        summary = summary.rename(
            columns={
                "scaler_name_": "scaler_name",
                "algorithm_": "algorithm",
                "n_clusters_": "n_clusters",
                "n_components_": "n_components",
            }
        )

        summary_sort = [
            column
            for column in [
                "test_silhouette_mean",
                "explained_variance_mean",
                "test_v_measure_mean",
                "test_nmi_mean",
                "test_davies_bouldin_mean",
            ]
            if column in summary.columns
        ]
        summary_ascending = [False, False, False, False, True][: len(summary_sort)]
        if summary_sort:
            summary = summary.sort_values(summary_sort, ascending=summary_ascending)

        self.holdout_summary_ = summary.reset_index(drop=True)
        return self.holdout_summary_.copy()

    def holdout_report(self, top_n=10):
        if self.holdout_summary_.empty:
            raise RuntimeError("Ejecuta evaluate_holdout() antes de consultar el holdout report.")
        return _decorate_holdout_table(self.holdout_summary_.head(top_n).copy())

    def search_influential_subsets(
        self,
        min_features=2,
        max_features=6,
        grouped=True,
        criterion="silhouette",
        include_holdout=False,
        test_size=0.25,
        n_splits=2,
        stratify_target=True,
        min_cluster_size=None,
        min_cluster_share=0.01,
    ):
        self._require_fit()
        ranked_variables = self.influential_variables(grouped=grouped, top_n=None, as_list=True)
        if not ranked_variables:
            raise ValueError("No hay variables influyentes disponibles para evaluar subconjuntos.")

        max_features = min(int(max_features), len(ranked_variables))
        min_features = max(2, int(min_features))
        if min_features > max_features:
            raise ValueError("min_features no puede ser mayor que max_features.")

        min_cluster_share = float(min_cluster_share)
        if not 0 <= min_cluster_share <= 1:
            raise ValueError("min_cluster_share debe estar entre 0 y 1.")
        if min_cluster_size is not None and int(min_cluster_size) < 1:
            raise ValueError("min_cluster_size debe ser None o un entero positivo.")

        criterion_map = {
            "silhouette": "silhouette",
            "davies_bouldin": "davies_bouldin",
            "holdout_silhouette": "test_silhouette_mean",
            "holdout_davies_bouldin": "test_davies_bouldin_mean",
        }
        criterion_column = criterion_map.get(criterion, criterion)
        if criterion_column.startswith("test_"):
            include_holdout = True

        data_frame = self._as_frame(self.data)
        n_samples = len(data_frame)
        share_based_min_size = max(1, int(np.ceil(n_samples * min_cluster_share)))
        required_min_cluster_size = max(share_based_min_size, 1 if min_cluster_size is None else int(min_cluster_size))
        target_arg = self.target if isinstance(self.target, str) else self.target_.copy() if self.target_ is not None else None

        self.influential_subset_search_ = pd.DataFrame()
        self.influential_subset_models_ = {}
        self.best_influential_subset_ = None
        self.best_influential_variables_ = []
        self.best_influential_fenrir_ = None

        results = []
        models = {}

        for n_features in range(min_features, max_features + 1):
            selected_variables = ranked_variables[:n_features]
            subset_model = Fenrir(
                data_frame,
                target=target_arg,
                features=selected_variables,
                exclude_columns=sorted(self.exclude_columns),
                variance_threshold=self.variance_threshold,
                cluster_range=self.cluster_range,
                algorithms=self.algorithms,
                scalers={name: clone(scaler) for name, scaler in self.scalers.items()},
                max_categories=self.max_categories,
                random_state=self.random_state,
            )
            subset_model.fit()
            size_report = subset_model.cluster_size_report()
            min_size = int(size_report["cluster_size"].min())
            max_size = int(size_report["cluster_size"].max())

            result_row: dict[str, Any] = {
                str(key): value for key, value in subset_model.best_configuration().to_dict().items()
            }
            result_row["n_selected_features"] = n_features
            result_row["selected_variables"] = selected_variables
            result_row["min_cluster_size"] = min_size
            result_row["max_cluster_size"] = max_size
            result_row["min_cluster_share"] = min_size / n_samples
            result_row["required_min_cluster_size"] = required_min_cluster_size
            result_row["passes_cluster_guard"] = min_size >= required_min_cluster_size

            if include_holdout:
                subset_model.evaluate_holdout(
                    test_size=test_size,
                    n_splits=n_splits,
                    stratify_target=stratify_target,
                )
                holdout_best = {
                    str(key): value
                    for key, value in subset_model.holdout_report(top_n=1).iloc[0].to_dict().items()
                }
                result_row.update(
                    {
                        key: value
                        for key, value in holdout_best.items()
                        if str(key).startswith("test_")
                    }
                )

            results.append(result_row)
            models[n_features] = subset_model

        results_df = pd.DataFrame(results)
        if criterion_column not in results_df.columns:
            raise ValueError(f"La columna de criterio '{criterion_column}' no esta disponible.")

        ascending = criterion_column.endswith("davies_bouldin") or criterion_column.endswith("davies_bouldin_mean")
        sort_columns = ["passes_cluster_guard", criterion_column, "n_selected_features"]
        sort_ascending = [False, ascending, True]
        if not ascending and "explained_variance" in results_df.columns:
            sort_columns.append("explained_variance")
            sort_ascending.append(False)

        results_df = results_df.sort_values(sort_columns, ascending=sort_ascending).reset_index(drop=True)
        results_df["selection_status"] = np.where(
            results_df["passes_cluster_guard"],
            "valido",
            "cluster minimo insuficiente",
        )

        self.influential_subset_search_ = results_df
        self.influential_subset_models_ = models
        valid_results = results_df.loc[results_df["passes_cluster_guard"]]
        if valid_results.empty:
            raise ValueError(
                "Ningun subconjunto cumple el tamano minimo de cluster "
                f"({required_min_cluster_size} filas). "
                "Consulta influential_subset_search_ para revisar los candidatos descartados."
            )
        self.best_influential_subset_ = valid_results.iloc[0].to_dict()
        self.best_influential_variables_ = list(self.best_influential_subset_["selected_variables"])
        self.best_influential_fenrir_ = models[int(self.best_influential_subset_["n_selected_features"])]
        return results_df.copy()

    def best_influential_variable_list(self):
        if not self.best_influential_variables_:
            raise RuntimeError("Ejecuta search_influential_subsets() antes de pedir la mejor lista compacta.")
        return list(self.best_influential_variables_)

    def best_influential_model(self):
        if self.best_influential_fenrir_ is None:
            raise RuntimeError("Ejecuta search_influential_subsets() antes de pedir el mejor modelo compacto.")
        return self.best_influential_fenrir_

    def evaluate_stability(
        self,
        data=None,
        sample_fraction=0.8,
        n_splits=3,
        stratify_target=True,
    ):
        self._require_fit()
        raw = self._as_frame(self.data if data is None else data)
        target_series = None
        target_name = self.target if isinstance(self.target, str) else None

        if isinstance(self.target, str) and self.target in raw.columns:
            target_series = raw[self.target].copy()
        elif self.target is not None:
            target_series = pd.Series(self.target, index=raw.index, name=self.target_name_)
            target_name = self.target_name_
            raw = raw.copy()
            raw[target_name] = target_series.values

        if not 0 < float(sample_fraction) < 1:
            raise ValueError("sample_fraction debe estar entre 0 y 1.")
        if int(n_splits) < 1:
            raise ValueError("n_splits debe ser al menos 1.")

        if stratify_target and target_series is not None and target_series.nunique(dropna=False) > 1:
            stratify_values = target_series.astype(str).where(target_series.notna(), "__missing__")
            splitter = StratifiedShuffleSplit(
                n_splits=int(n_splits),
                train_size=float(sample_fraction),
                random_state=self.random_state,
            )
            iterator = splitter.split(raw, stratify_values)
        else:
            splitter = ShuffleSplit(
                n_splits=int(n_splits),
                train_size=float(sample_fraction),
                random_state=self.random_state,
            )
            iterator = splitter.split(raw)

        base_result = self.best_configuration()
        base_labels = self._require_best_labels().copy() if data is None else self.predict(raw)
        results = []

        for split_id, (sample_idx, _) in enumerate(iterator, start=1):
            sample_raw = raw.iloc[sample_idx].copy()
            evaluator = self._new_like(sample_raw, target_name=target_name)
            evaluator.fit()

            reference_labels = base_labels.loc[sample_raw.index] if data is None else self.predict(sample_raw)
            candidate_labels = evaluator._require_best_labels().reindex(sample_raw.index)
            candidate_result = evaluator.best_configuration()
            size_report = evaluator.cluster_size_report()

            row = {
                "split": split_id,
                "sample_size": len(sample_raw),
                "sample_fraction": len(sample_raw) / len(raw),
                "scaler_name": candidate_result["scaler_name"],
                "algorithm": candidate_result["algorithm"],
                "n_clusters": int(candidate_result["n_clusters"]),
                "n_components": int(candidate_result["n_components"]),
                "silhouette": float(candidate_result["silhouette"]),
                "explained_variance": float(candidate_result["explained_variance"]),
                "davies_bouldin": float(candidate_result["davies_bouldin"]),
                "min_cluster_size": int(size_report["cluster_size"].min()),
                "same_scaler": candidate_result["scaler_name"] == base_result["scaler_name"],
                "same_algorithm": candidate_result["algorithm"] == base_result["algorithm"],
                "same_n_clusters": int(candidate_result["n_clusters"]) == int(base_result["n_clusters"]),
                "same_n_components": int(candidate_result["n_components"]) == int(base_result["n_components"]),
            }

            mask = reference_labels.notna() & candidate_labels.notna()
            if mask.sum() >= 2 and reference_labels[mask].nunique() > 1 and candidate_labels[mask].nunique() > 1:
                row["ari_vs_base"] = float(adjusted_rand_score(reference_labels[mask], candidate_labels[mask]))
                row["nmi_vs_base"] = float(normalized_mutual_info_score(reference_labels[mask], candidate_labels[mask]))
                row["v_measure_vs_base"] = float(v_measure_score(reference_labels[mask], candidate_labels[mask]))
            else:
                row["ari_vs_base"] = np.nan
                row["nmi_vs_base"] = np.nan
                row["v_measure_vs_base"] = np.nan

            results.append(row)

        if not results:
            raise RuntimeError("No se pudo calcular la estabilidad del clustering.")

        results_df = pd.DataFrame(results)
        sort_columns = [
            column
            for column in ["ari_vs_base", "nmi_vs_base", "v_measure_vs_base", "silhouette", "min_cluster_size"]
            if column in results_df.columns
        ]
        sort_ascending = [False, False, False, False, False][: len(sort_columns)]
        self.stability_results_ = results_df.sort_values(sort_columns, ascending=sort_ascending).reset_index(drop=True)

        same_configuration_rate = (
            self.stability_results_[["same_scaler", "same_algorithm", "same_n_clusters", "same_n_components"]]
            .all(axis=1)
            .mean()
        )
        ari_mean = self.stability_results_["ari_vs_base"].mean()
        summary = pd.DataFrame(
            [
                {
                    "n_splits": int(n_splits),
                    "sample_fraction": float(sample_fraction),
                    "ari_vs_base_mean": ari_mean,
                    "ari_vs_base_std": self.stability_results_["ari_vs_base"].std(),
                    "nmi_vs_base_mean": self.stability_results_["nmi_vs_base"].mean(),
                    "nmi_vs_base_std": self.stability_results_["nmi_vs_base"].std(),
                    "v_measure_vs_base_mean": self.stability_results_["v_measure_vs_base"].mean(),
                    "v_measure_vs_base_std": self.stability_results_["v_measure_vs_base"].std(),
                    "same_scaler_rate": self.stability_results_["same_scaler"].mean(),
                    "same_algorithm_rate": self.stability_results_["same_algorithm"].mean(),
                    "same_n_clusters_rate": self.stability_results_["same_n_clusters"].mean(),
                    "same_n_components_rate": self.stability_results_["same_n_components"].mean(),
                    "same_configuration_rate": same_configuration_rate,
                    "silhouette_mean": self.stability_results_["silhouette"].mean(),
                    "min_cluster_size_mean": self.stability_results_["min_cluster_size"].mean(),
                    "stability_reading": _interpret_stability(ari_mean),
                }
            ]
        )
        self.stability_summary_ = summary
        return summary.copy()

    def stability_report(self, top_n=5):
        if self.stability_results_.empty:
            raise RuntimeError("Ejecuta evaluate_stability() antes de consultar el stability report.")
        return _decorate_stability_table(self.stability_results_.head(top_n).copy())

    def analysis_report(
        self,
        top_n=5,
        leaderboard_top_n=6,
        holdout_top_n=5,
        compact_max_features=5,
        compact_criterion="holdout_silhouette",
        compact_include_holdout=None,
        test_size=0.25,
        n_splits=2,
        stratify_target=True,
        min_cluster_share=0.01,
        include_stability=True,
        stability_sample_fraction=0.8,
        stability_n_splits=3,
        stability_top_n=5,
        stability_stratify_target=True,
    ):
        self._require_fit()
        full_influential_list = self.list_influential_variables(top_n=None)

        if compact_include_holdout is None:
            compact_include_holdout = str(compact_criterion).startswith("holdout_")

        holdout_summary = self.evaluate_holdout(
            test_size=test_size,
            n_splits=n_splits,
            stratify_target=stratify_target,
        )
        holdout_top = self.holdout_report(top_n=holdout_top_n)
        holdout_summary = _decorate_holdout_table(holdout_summary.copy())

        subset_search = self.search_influential_subsets(
            min_features=2,
            max_features=min(int(compact_max_features), len(full_influential_list)),
            criterion=compact_criterion,
            include_holdout=compact_include_holdout,
            test_size=test_size,
            n_splits=n_splits,
            stratify_target=stratify_target,
            min_cluster_share=min_cluster_share,
        )
        best_compact_variables = self.best_influential_variable_list()
        compact_model = self.best_influential_model()

        report = {
            "best_configuration": self.best_configuration().copy(),
            "metric_report": self.metric_report().copy(),
            "preprocessing_report": self.preprocessing_report().copy(),
            "scaler_report": self.scaler_report().copy(),
            "leaderboard": self.summary(top_n=leaderboard_top_n).copy(),
            "influential_variables": self.list_influential_variables(top_n=top_n),
            "cluster_size_report": self.cluster_size_report().copy(),
            "cluster_groupby": self.cluster_groupby(top_n=top_n).copy(),
            "cluster_target_report": self.cluster_target_report(normalize="index").copy() if self.target_ is not None else None,
            "holdout_summary": holdout_summary.copy(),
            "holdout_top": holdout_top.copy(),
            "compact_subset_search": subset_search.copy(),
            "best_compact_variables": list(best_compact_variables),
            "best_compact_configuration": compact_model.best_configuration().copy(),
            "best_compact_groupby": compact_model.cluster_groupby(top_n=min(top_n, len(best_compact_variables))).copy(),
        }

        if include_stability:
            report["stability_summary"] = _decorate_stability_summary_table(
                self.evaluate_stability(
                    sample_fraction=stability_sample_fraction,
                    n_splits=stability_n_splits,
                    stratify_target=stability_stratify_target,
                )
            )
            report["stability_top"] = self.stability_report(top_n=stability_top_n)
        else:
            report["stability_summary"] = None
            report["stability_top"] = None

        self.analysis_report_ = report
        return report

    def plot_scree(self, scaler_name=None, max_components=10, ax=None):
        plt_module = _require_matplotlib()
        profile = self._profile_for(scaler_name=scaler_name)
        report = profile["component_report"].head(max_components)

        if ax is None:
            _, ax = plt_module.subplots(figsize=(8, 4))

        x_values = report["component"]
        ax.bar(
            x_values,
            report["explained_variance"],
            color="#355070",
            alpha=0.8,
            label="Varianza explicada",
        )
        ax.plot(
            x_values,
            report["cumulative_variance"],
            color="#e56b6f",
            marker="o",
            linewidth=2,
            label="Varianza acumulada",
        )
        ax.axhline(
            self.variance_threshold,
            color="#6d597a",
            linestyle="--",
            label=f"Umbral {self.variance_threshold:.0%}",
        )
        ax.axvline(
            int(profile["selected_components"]),
            color="#b56576",
            linestyle=":",
            label=f"PC elegidas: {int(profile['selected_components'])}",
        )
        ax.set_xlabel("Componente principal")
        ax.set_ylabel("Proporcion de varianza")
        ax.set_title(f"Scree plot con {profile['scaler_name']}")
        ax.set_xticks(x_values)
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False)
        return ax

    def plot_correlation_circle(self, scaler_name=None, components=(1, 2), top_n=12, ax=None):
        plt_module = _require_matplotlib()
        profile = self._profile_for(scaler_name=scaler_name)
        component_numbers, component_names = self._resolve_components(profile, components)

        if len(component_names) != 2 or component_names[0] == component_names[1]:
            raise ValueError("El circulo de correlaciones necesita exactamente dos componentes distintas.")

        selected_features = self.feature_report(
            scaler_name=scaler_name,
            components=component_numbers,
            top_n=top_n,
        ).index
        coords = profile["loadings"].loc[selected_features, component_names]

        if ax is None:
            _, ax = plt_module.subplots(figsize=(7, 7))

        circle = plt_module.Circle((0, 0), 1.0, fill=False, linestyle="--", color="#adb5bd")
        ax.add_artist(circle)

        for feature, row in coords.iterrows():
            ax.arrow(
                0,
                0,
                row[component_names[0]],
                row[component_names[1]],
                width=0.003,
                head_width=0.04,
                length_includes_head=True,
                color="#355070",
                alpha=0.7,
            )
            ax.text(
                row[component_names[0]] * 1.08,
                row[component_names[1]] * 1.08,
                feature,
                fontsize=9,
            )

        ax.axhline(0, color="#adb5bd", linewidth=1)
        ax.axvline(0, color="#adb5bd", linewidth=1)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(
            f"{component_names[0]} ({profile['component_report'].loc[int(component_numbers[0]) - 1, 'explained_variance']:.1%})"
        )
        ax.set_ylabel(
            f"{component_names[1]} ({profile['component_report'].loc[int(component_numbers[1]) - 1, 'explained_variance']:.1%})"
        )
        ax.set_title("Circulo de correlaciones")
        return ax

    def plot_feature_contributions(
        self,
        scaler_name=None,
        components=None,
        top_n=10,
        grouped=True,
        ax=None,
    ):
        plt_module = _require_matplotlib()
        report = self.influential_variables(
            scaler_name=scaler_name,
            components=components,
            top_n=top_n,
            grouped=grouped,
            as_list=False,
        )
        values = pd.Series(report["weighted_contribution"], dtype="float64").sort_values()

        if ax is None:
            _, ax = plt_module.subplots(figsize=(8, 5))

        ax.barh(values.index, np.asarray(values, dtype=float), color="#355070", alpha=0.9)
        ax.set_xlabel("Contribucion ponderada")
        ax.set_title("Variables mas influyentes")
        return ax

    def plot_feature_quality(
        self,
        scaler_name=None,
        components=None,
        top_n=10,
        grouped=True,
        ax=None,
    ):
        plt_module = _require_matplotlib()
        report = self.influential_variables(
            scaler_name=scaler_name,
            components=components,
            top_n=top_n,
            grouped=grouped,
            as_list=False,
        )
        values = pd.Series(report["weighted_cos2"], dtype="float64").sort_values()

        if ax is None:
            _, ax = plt_module.subplots(figsize=(8, 5))

        ax.barh(values.index, np.asarray(values, dtype=float), color="#e56b6f", alpha=0.9)
        ax.set_xlabel("Calidad de representacion ponderada")
        ax.set_title("Calidad de representacion de variables")
        return ax

    def plot_clusters(self, scaler_name=None, ax=None):
        plt_module = _require_matplotlib()
        result = self._result_for_scaler(scaler_name=scaler_name)
        profile = self._profile_for(scaler_name=result["scaler_name"])
        key = (
            result["scaler_name"],
            result["algorithm"],
            int(result["n_clusters"]),
            int(result["n_components"]),
        )
        labels = self.cluster_labels_[key]
        projection = profile["scores"].iloc[:, :2]

        if ax is None:
            _, ax = plt_module.subplots(figsize=(7, 5))

        scatter = ax.scatter(
            projection.iloc[:, 0],
            projection.iloc[:, 1],
            c=labels,
            cmap="tab10",
            s=28,
            alpha=0.85,
        )
        ax.set_xlabel(f"PC1 ({profile['component_report'].loc[0, 'explained_variance']:.1%})")
        ax.set_ylabel(f"PC2 ({profile['component_report'].loc[1, 'explained_variance']:.1%})")
        ax.set_title(f"Clusters con {result['algorithm']} y {result['scaler_name']}")
        legend = ax.legend(*scatter.legend_elements(), title="Cluster", frameon=False, loc="best")
        ax.add_artist(legend)
        return ax
