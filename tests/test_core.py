from __future__ import annotations

from fenrir import Fenrir
from sklearn.preprocessing import StandardScaler


def test_fit_predict_and_reports(fitted_model, sample_frame):
    labels = fitted_model.fit_predict()

    assert len(labels) == len(sample_frame)
    assert labels.name == "cluster"

    best = fitted_model.best_configuration()
    assert best["algorithm"] == "kmeans"
    assert int(best["n_clusters"]) >= 2

    metric_report = fitted_model.metric_report()
    assert {"silhouette", "davies_bouldin"}.issubset(set(metric_report["metric"]))

    grouped = fitted_model.cluster_groupby(top_n=4)
    assert "cluster_size" in grouped.columns
    assert grouped.shape[0] >= 2


def test_analysis_report_and_predict_on_new_frame(sample_frame):
    model = Fenrir(
        sample_frame,
        target="target",
        variance_threshold=0.85,
        cluster_range=range(3, 4),
        algorithms=("kmeans",),
        scalers={"StandardScaler": StandardScaler()},
        random_state=21,
    ).fit()

    report = model.analysis_report(
        top_n=5,
        leaderboard_top_n=5,
        holdout_top_n=5,
        compact_max_features=4,
        test_size=0.2,
        n_splits=1,
        include_stability=True,
        stability_n_splits=1,
        stability_top_n=5,
    )

    assert not report["holdout_top"].empty
    assert report["best_compact_variables"]
    assert report["best_compact_configuration"].shape[0] >= 1

    predicted = model.predict(sample_frame.drop(columns=["target"]))
    assert predicted.name == "cluster"
    assert len(predicted) == len(sample_frame)
