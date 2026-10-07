"""Traspaso Bahamut -> Fenrir: los cortes los decide Bahamut, Fenrir los respeta."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from fenrir import Fenrir

# Bahamut se usa instalado. Como respaldo, se acepta un clon hermano junto a
# este repositorio; nunca una ruta absoluta de una maquina concreta.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _candidate in (_REPO_ROOT.parent / "bahamut" / "src", _REPO_ROOT.parent.parent / "bahamut" / "src"):
    if (_candidate / "bahamut" / "__init__.py").is_file():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

bahamut = pytest.importorskip(
    "bahamut",
    reason="bahamut no esta instalado ni hay un clon hermano junto a este repositorio",
)
BahamutSplit = bahamut.BahamutSplit


MODEL_KWARGS = dict(
    cluster_range=range(2, 5),
    algorithms=("kmeans",),
    scalers={"StandardScaler": StandardScaler()},
    random_state=4,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 150
    grupo = rng.integers(0, 3, n)
    # Solapamiento deliberado: si los clusters cayeran perfectos, las metricas
    # externas valdrian 1.0 en los tres bloques y el test no distinguiria un
    # traspaso correcto de uno que reajusta por debajo.
    return pd.DataFrame(
        {
            "x1": rng.normal(grupo * 2.2, 1.6),
            "x2": rng.normal(grupo * 1.8, 1.6),
            "x3": rng.normal(0, 1, n),
            "canal": np.where(grupo == 0, "web", np.where(grupo == 1, "tienda", "partner")),
            "id_cliente": np.arange(n) // 5,
            "segmento": pd.Series(grupo).map({0: "rojo", 1: "azul", 2: "verde"}),
        }
    )


@pytest.fixture
def segmentos(frame: pd.DataFrame) -> dict:
    splitter = (
        BahamutSplit(frame)
        .definir_problema("clasificacion")
        .definir_objetivo("segmento")
        .definir_predictoras(exclude_cols=["id_cliente"])
        .definir_estratificacion(modo="target")
        .configurar_split(test_size=0.2, validation_size=0.2, shuffle=True, random_state=11)
    )
    return splitter.ejecutar_segmentacion()


# --------------------------------------------------------------------------
# Construccion
# --------------------------------------------------------------------------

def test_el_ajuste_usa_solo_el_bloque_de_entrenamiento(segmentos):
    model = Fenrir.from_bahamut(segmentos, fit=True, **MODEL_KWARGS)

    assert len(model.data) == len(segmentos["X_train"])
    assert len(model.best_labels_) == len(segmentos["X_train"])
    assert set(model.bahamut_frames_) == {"train", "validation", "test"}


def test_el_objetivo_se_deduce_de_y_train(segmentos):
    model = Fenrir.from_bahamut(segmentos, **MODEL_KWARGS)

    assert model.target == "segmento"
    assert "segmento" in model.bahamut_frames_["train"].columns


def test_df_original_rehidrata_las_columnas_que_no_viajan_en_el_bundle(frame, segmentos):
    assert "id_cliente" not in segmentos["X_train"].columns

    model = Fenrir.from_bahamut(segmentos, df=frame, exclude_columns=["id_cliente"], **MODEL_KWARGS)

    assert "id_cliente" in model.bahamut_frames_["train"].columns
    assert len(model.bahamut_frames_["test"]) == len(segmentos["X_test"])


def test_los_bloques_conservan_los_indices_de_bahamut(segmentos):
    model = Fenrir.from_bahamut(segmentos, **MODEL_KWARGS)

    for split_name, index_key in [
        ("train", "indices_train"),
        ("validation", "indices_validation"),
        ("test", "indices_test"),
    ]:
        assert list(model.bahamut_frames_[split_name].index) == list(segmentos[index_key])


# --------------------------------------------------------------------------
# Evaluacion fuera de muestra
# --------------------------------------------------------------------------

def test_evaluate_bahamut_splits_puntua_los_tres_bloques(segmentos):
    model = Fenrir.from_bahamut(segmentos, fit=True, **MODEL_KWARGS)

    resultados = model.evaluate_bahamut_splits()

    assert list(resultados["split"]) == ["train", "validation", "test"]
    assert list(resultados["origen"]) == ["ajuste", "fuera de muestra", "fuera de muestra"]
    assert list(resultados["n_filas"]) == [
        len(segmentos["X_train"]),
        len(segmentos["X_validation"]),
        len(segmentos["X_test"]),
    ]
    assert resultados["silhouette"].notna().all()
    assert resultados["v_measure"].notna().all()


def test_no_reajusta_nada_al_evaluar(segmentos):
    model = Fenrir.from_bahamut(segmentos, fit=True, **MODEL_KWARGS)
    configuracion_previa = model.best_configuration().to_dict()
    etiquetas_previas = model.best_labels_.copy()

    model.evaluate_bahamut_splits()

    assert model.best_configuration().to_dict() == configuracion_previa
    pd.testing.assert_series_equal(model.best_labels_, etiquetas_previas)


def test_la_fila_de_train_coincide_con_las_metricas_del_ajuste(segmentos):
    model = Fenrir.from_bahamut(segmentos, fit=True, **MODEL_KWARGS)

    fila_train = model.evaluate_bahamut_splits().set_index("split").loc["train"]

    assert fila_train["silhouette"] == pytest.approx(float(model.best_configuration()["silhouette"]))
    assert int(fila_train["n_clusters_observados"]) == int(model.best_configuration()["n_clusters"])


def test_include_train_false_deja_solo_los_bloques_no_vistos(segmentos):
    model = Fenrir.from_bahamut(segmentos, fit=True, **MODEL_KWARGS)

    resultados = model.evaluate_bahamut_splits(include_train=False)

    assert list(resultados["split"]) == ["validation", "test"]


def test_agglomerative_se_evalua_fuera_de_muestra_aunque_no_admita_predict(segmentos):
    model = Fenrir.from_bahamut(
        segmentos,
        fit=True,
        cluster_range=range(3, 4),
        algorithms=("agglomerative",),
        scalers={"StandardScaler": StandardScaler()},
        random_state=4,
    )

    with pytest.raises(ValueError, match="agglomerative"):
        model.predict(model.bahamut_frames_["test"])

    resultados = model.evaluate_bahamut_splits()

    assert len(resultados) == 3
    assert resultados.loc[resultados["split"] == "test", "silhouette"].notna().all()


def test_bundle_sin_target_solo_reporta_metricas_internas(frame):
    splitter = (
        BahamutSplit(frame.drop(columns=["segmento"]))
        .definir_problema("clusterizacion")
        .definir_objetivo(None)
        .definir_predictoras(exclude_cols=["id_cliente"])
        .definir_estratificacion(modo="ninguna")
        .configurar_split(test_size=0.25, shuffle=True, random_state=5)
    )
    model = Fenrir.from_bahamut(splitter.ejecutar_segmentacion(), fit=True, **MODEL_KWARGS)

    resultados = model.evaluate_bahamut_splits()

    assert list(resultados["split"]) == ["train", "test"]
    assert resultados["silhouette"].notna().all()
    assert resultados["ari"].isna().all()


# --------------------------------------------------------------------------
# Lectura interpretada
# --------------------------------------------------------------------------

def test_el_report_anade_la_lectura_interpretada(segmentos):
    model = Fenrir.from_bahamut(segmentos, fit=True, **MODEL_KWARGS)
    model.evaluate_bahamut_splits()

    reporte = model.bahamut_split_report()

    assert list(reporte.columns)[:3] == ["split", "origen", "n_filas"]
    assert {"silhouette_reading", "davies_bouldin_reading", "cluster_balance"}.issubset(reporte.columns)
    assert reporte["silhouette_reading"].map(lambda value: isinstance(value, str)).all()


# --------------------------------------------------------------------------
# Errores
# --------------------------------------------------------------------------

def test_una_entrada_que_no_es_un_bundle_falla_con_un_mensaje_util():
    with pytest.raises(TypeError, match="no parece un bundle de Bahamut"):
        Fenrir.from_bahamut({"a": 1})


def test_multi_target_exige_elegir_columna(frame):
    splitter = (
        BahamutSplit(frame)
        .definir_problema("clasificacion")
        .definir_objetivo(["segmento", "canal"])
        .definir_predictoras(exclude_cols=["id_cliente"])
        .definir_estratificacion(modo="ninguna")
        .configurar_split(test_size=0.25, shuffle=True, random_state=2)
    )
    segmentos = splitter.ejecutar_segmentacion()

    with pytest.raises(ValueError, match="varias columnas objetivo"):
        Fenrir.from_bahamut(segmentos)

    model = Fenrir.from_bahamut(segmentos, target="segmento", **MODEL_KWARGS)
    assert model.target == "segmento"
    model.fit()
    assert "canal" not in model.raw_training_frame_.columns


def test_rehidratar_no_reintroduce_predictoras_excluidas(frame, segmentos):
    model = Fenrir.from_bahamut(segmentos, df=frame, fit=True, **MODEL_KWARGS)
    assert "id_cliente" in model.bahamut_frames_["train"].columns
    assert "id_cliente" not in model.raw_training_frame_.columns


def test_rehidratar_rechaza_indice_ambiguo(frame, segmentos):
    ambiguous = pd.concat([frame, frame.iloc[:1]])
    with pytest.raises(ValueError, match="indice unico"):
        Fenrir.from_bahamut(segmentos, df=ambiguous)


def test_bundle_con_indice_repetido_no_multiplica_filas():
    x = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6]}, index=[0, 0, 1])
    y = pd.Series(["a", "b", "c"], index=x.index, name="target")
    model = Fenrir.from_bahamut({"X_train": x, "y_train": y, "indices_train": x.index})
    assert len(model.data) == 3
    assert model.data["target"].tolist() == ["a", "b", "c"]


def test_bundle_rechaza_objetivo_desalineado():
    x = pd.DataFrame({"x1": [1, 2], "x2": [4, 5]}, index=[0, 1])
    y = pd.Series(["a", "b"], index=[1, 0], name="target")
    with pytest.raises(ValueError, match="indices de X e y"):
        Fenrir.from_bahamut({"X_train": x, "y_train": y, "indices_train": x.index})


def test_evaluar_sin_ajustar_falla(segmentos):
    with pytest.raises(RuntimeError, match="fit()"):
        Fenrir.from_bahamut(segmentos, **MODEL_KWARGS).evaluate_bahamut_splits()


def test_una_instancia_sin_bundle_explica_como_conseguir_uno(frame):
    model = Fenrir(frame.drop(columns=["segmento", "canal"]), **MODEL_KWARGS).fit()

    with pytest.raises(RuntimeError, match="from_bahamut"):
        model.evaluate_bahamut_splits()
