"""Regresiones de los defectos detectados en la auditoria de 2026."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from fenrir import Fenrir


def build_frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "a": np.concatenate([rng.normal(0, 1, n // 2), rng.normal(8, 1, n // 2)]),
            "b": np.concatenate([rng.normal(0, 1, n // 2), rng.normal(8, 1, n // 2)]),
            "canal": ["web"] * (n // 2) + ["tienda"] * (n // 2),
        }
    )
    frame["t"] = [0] * (n // 2) + [1] * (n // 2)
    return frame


def fit_model(frame: pd.DataFrame, **overrides) -> Fenrir:
    kwargs = dict(
        target="t",
        cluster_range=range(2, 3),
        algorithms=("kmeans",),
        scalers={"StandardScaler": StandardScaler()},
        random_state=5,
    )
    kwargs.update(overrides)
    return Fenrir(frame, **kwargs).fit()


def test_compact_search_rejects_all_candidates_below_minimum_cluster_size():
    frame = build_frame(n=20)
    model = fit_model(frame)

    with pytest.raises(ValueError, match="Ningun subconjunto cumple"):
        model.search_influential_subsets(max_features=2, min_cluster_size=len(frame))

    assert not model.influential_subset_search_.empty
    assert not model.influential_subset_search_["passes_cluster_guard"].any()
    assert model.best_influential_subset_ is None
    assert model.best_influential_variables_ == []
    assert model.best_influential_fenrir_ is None


def test_failed_compact_search_clears_previous_best_model():
    frame = build_frame(n=20)
    model = fit_model(frame)
    model.search_influential_subsets(max_features=2, min_cluster_share=0)
    assert model.best_influential_model() is not None

    with pytest.raises(ValueError, match="Ningun subconjunto cumple"):
        model.search_influential_subsets(max_features=2, min_cluster_share=1)

    with pytest.raises(RuntimeError, match="search_influential_subsets"):
        model.best_influential_model()
    with pytest.raises(RuntimeError, match="search_influential_subsets"):
        model.best_influential_variable_list()


# --------------------------------------------------------------------------
# Esquema de prediccion
# --------------------------------------------------------------------------

def test_predict_con_columna_de_entrenamiento_ausente_falla():
    frame = build_frame()
    model = fit_model(frame)

    with pytest.raises(ValueError, match="faltan columnas usadas en el ajuste"):
        model.predict(frame.drop(columns=["t", "b"]))


def test_predict_con_columna_ausente_avisa_si_se_desactiva_el_modo_estricto():
    frame = build_frame()
    model = fit_model(frame, strict_predict_schema=False)

    with pytest.warns(UserWarning, match="faltan columnas usadas en el ajuste"):
        etiquetas = model.predict(frame.drop(columns=["t", "b"]))

    assert len(etiquetas) == len(frame)


def test_predict_ignora_columnas_extra():
    frame = build_frame()
    model = fit_model(frame)

    base = model.predict(frame.drop(columns=["t"]))
    con_extra = model.predict(frame.drop(columns=["t"]).assign(irrelevante=1.0))

    pd.testing.assert_series_equal(base, con_extra)


def test_nivel_categorico_ausente_se_codifica_como_cero_no_como_mediana():
    frame = build_frame()
    model = fit_model(frame)

    solo_web = frame.loc[frame["canal"] == "web"].drop(columns=["t"]).copy()
    preparado = model._prepare_frame(solo_web, fit=False)

    columnas_tienda = [
        column
        for column in model.feature_names_
        if model.feature_source_map_.get(column) == "canal" and "tienda" in column
    ]
    assert columnas_tienda, "el ajuste deberia haber codificado el nivel 'tienda'"
    for column in columnas_tienda:
        assert (preparado[column] == 0).all()


# --------------------------------------------------------------------------
# Candidatos descartados durante fit()
# --------------------------------------------------------------------------

def test_los_candidatos_fallidos_quedan_registrados():
    frame = build_frame(n=20)
    model = Fenrir(
        frame,
        target="t",
        cluster_range=range(2, 25),
        algorithms=("kmeans",),
        scalers={"StandardScaler": StandardScaler()},
        random_state=5,
    ).fit()

    reporte = model.search_failures_report()

    assert not model.search_results_.empty
    assert set(reporte.columns) >= {"algorithm", "n_clusters", "error", "detail"}


def test_sin_ningun_candidato_valido_el_error_explica_los_motivos(monkeypatch):
    frame = build_frame(n=20)

    def _siempre_falla(self, algorithm, n_clusters, reduced_scores):
        raise ValueError("configuracion invalida de prueba")

    monkeypatch.setattr(Fenrir, "_fit_candidate", _siempre_falla)

    with pytest.raises(RuntimeError) as error:
        Fenrir(
            frame,
            target="t",
            cluster_range=range(2, 4),
            algorithms=("kmeans",),
            scalers={"StandardScaler": StandardScaler()},
        ).fit()

    mensaje = str(error.value)
    assert "Motivos:" in mensaje
    assert "configuracion invalida de prueba" in mensaje


# --------------------------------------------------------------------------
# Exportacion
# --------------------------------------------------------------------------

def test_las_hojas_del_excel_ejecutivo_son_slugs_estables(tmp_path):
    frame = build_frame()
    model = fit_model(frame)

    ruta = model.export_executive_workbook(
        file_name=tmp_path / "ejecutivo.xlsx",
        analysis_config={
            "top_n": 4,
            "leaderboard_top_n": 4,
            "holdout_top_n": 4,
            "compact_max_features": 2,
            "test_size": 0.25,
            "n_splits": 1,
            "include_stability": False,
        },
    )

    with pd.ExcelFile(ruta) as workbook:
        hojas = set(workbook.sheet_names)

    assert {
        "resumen_ejecutivo",
        "configuracion",
        "leaderboard",
        "metricas",
        "holdout_top",
        "compactos",
    }.issubset(hojas)
    assert all(hoja == hoja.lower() and " " not in hoja for hoja in hojas)
    assert all(len(hoja) <= 31 for hoja in hojas)


def test_las_exportaciones_relativas_aterrizan_en_el_directorio_de_trabajo(tmp_path, monkeypatch):
    from fenrir.interactive import _normalize_export_path, default_export_dir

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FENRIR_EXPORT_DIR", raising=False)

    assert default_export_dir() == tmp_path
    assert _normalize_export_path("", "fenrir_export", "xlsx").parent == tmp_path
    assert _normalize_export_path("salida.csv", "fenrir_export", "csv") == tmp_path / "salida.csv"


def test_la_variable_de_entorno_puede_redirigir_las_exportaciones(tmp_path, monkeypatch):
    from fenrir.interactive import _normalize_export_path

    destino = tmp_path / "informes"
    monkeypatch.setenv("FENRIR_EXPORT_DIR", str(destino))

    assert _normalize_export_path("x.json", "fenrir_export", "json").parent == destino


# --------------------------------------------------------------------------
# Plumbing compartido
# --------------------------------------------------------------------------

def test_el_nucleo_no_importa_el_stack_de_notebook():
    import fenrir.core as core_module

    with open(core_module.__file__, encoding="utf-8") as handle:
        contenido = handle.read()

    assert "import ipywidgets" not in contenido
    assert "from IPython" not in contenido


def test_la_capa_interactiva_usa_el_modulo_compartido():
    import fenrir.interactive as interactive_module

    with open(interactive_module.__file__, encoding="utf-8") as handle:
        contenido = handle.read()

    assert "from ._notebook import" in contenido
    assert "import ipywidgets as widgets" not in contenido
