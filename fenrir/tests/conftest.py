from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent

if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from fenrir import Fenrir


@pytest.fixture
def sample_frame():
    values, labels = make_blobs(
        n_samples=60,
        centers=3,
        n_features=4,
        cluster_std=0.55,
        random_state=7,
    )
    frame = pd.DataFrame(values, columns=["x1", "x2", "x3", "x4"])
    frame["segmento"] = pd.Series(labels).map({0: "rojo", 1: "azul", 2: "verde"})
    frame["canal"] = np.where(labels == 0, "web", np.where(labels == 1, "store", "partner"))
    frame["target"] = labels
    return frame


@pytest.fixture
def fitted_model(sample_frame):
    model = Fenrir(
        sample_frame,
        target="target",
        variance_threshold=0.85,
        cluster_range=range(3, 4),
        algorithms=("kmeans",),
        scalers={"StandardScaler": StandardScaler()},
        random_state=13,
    )
    model.fit()
    return model
