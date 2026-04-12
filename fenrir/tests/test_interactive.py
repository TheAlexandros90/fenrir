from __future__ import annotations

import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from fenrir import Fenrir


def _current_fenrir_config(model):
    return {
        "target": model.target,
        "features": None if model.features is None else list(model.features),
        "exclude_columns": sorted(model.exclude_columns),
        "variance_threshold": model.variance_threshold,
        "cluster_range": tuple(model.cluster_range),
        "algorithms": tuple(model.algorithms),
        "scalers": tuple(model.scalers.keys()),
        "max_categories": model.max_categories,
        "random_state": model.random_state,
    }


def _analysis_config():
    return {
        "top_n": 5,
        "leaderboard_top_n": 5,
        "holdout_top_n": 5,
        "compact_max_features": 4,
        "compact_criterion": "holdout_silhouette",
        "test_size": 0.2,
        "n_splits": 1,
        "min_cluster_share": 0.01,
        "include_stability": True,
        "stability_sample_fraction": 0.75,
        "stability_n_splits": 1,
        "stability_top_n": 5,
    }


def test_presets_json_and_executive_export(fitted_model, sample_frame, tmp_path):
    fenrir_config = _current_fenrir_config(fitted_model)
    analysis_config = _analysis_config()

    fitted_model.save_workbench_preset("base", fenrir_config, analysis_config)
    assert fitted_model.list_workbench_presets() == ["base"]

    json_path = fitted_model.save_workbench_presets_json(tmp_path / "presets.json")
    assert json_path.exists()

    other_model = Fenrir(
        sample_frame,
        target="target",
        variance_threshold=0.85,
        cluster_range=range(3, 4),
        algorithms=("kmeans",),
        scalers={"StandardScaler": StandardScaler()},
        random_state=9,
    ).fit()
    loaded_store, loaded_path = other_model.load_workbench_presets_json(tmp_path / "presets.json")

    assert loaded_path.exists()
    assert "base" in loaded_store

    workbook_path = fitted_model.export_executive_workbook(
        file_name=tmp_path / "fenrir_executivo.xlsx",
        analysis_config=analysis_config,
        fenrir_config=fenrir_config,
    )
    assert workbook_path.exists()

    with pd.ExcelFile(workbook_path) as workbook:
        sheet_names = set(workbook.sheet_names)

    assert {
        "resumen_ejecutivo",
        "configuracion",
        "leaderboard",
        "metricas",
        "holdout_top",
        "compactos",
    }.issubset(sheet_names)


def test_compare_candidates_and_widget_smoke(fitted_model):
    widgets = pytest.importorskip("ipywidgets")

    current_fenrir = _current_fenrir_config(fitted_model)
    analysis_config = _analysis_config()
    preset_fenrir = dict(current_fenrir)
    preset_fenrir["features"] = ["x1", "x2", "x3", "segmento", "canal"]

    fitted_model.save_workbench_preset("compacto", preset_fenrir, analysis_config)
    comparison = fitted_model.compare_workbench_candidates(
        "__current__",
        "compacto",
        current_fenrir_config=current_fenrir,
        current_analysis_config=analysis_config,
    )

    assert "coincide" in comparison["configuration"].columns
    assert not comparison["quality"].empty
    assert not comparison["validation"].empty

    pca_panel = fitted_model.interactive_pca(top_n=5)
    clustering_panel = fitted_model.interactive_clustering(
        top_n=5,
        compact_max_features=4,
        test_size=0.2,
        n_splits=1,
        stability_n_splits=1,
    )
    table_tabs = fitted_model.interactive_tables(
        pca_top_n=5,
        cluster_top_n=5,
        compact_max_features=4,
        test_size=0.2,
        n_splits=1,
        stability_n_splits=1,
    )
    workbench = fitted_model.interactive_workbench(
        fenrir_config=current_fenrir,
        analysis_config=analysis_config,
    )

    assert isinstance(pca_panel, widgets.Widget)
    assert isinstance(clustering_panel, widgets.Widget)
    assert isinstance(table_tabs, widgets.Widget)
    assert isinstance(workbench, widgets.Widget)
