from pathlib import Path
import sys

import pandas as pd

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from fenrir import Fenrir


def main():
    df = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 40, 41, 42, 43],
            "x2": [1, 1, 2, 2, 8, 9, 8, 9],
            "x3": [10, 11, 10, 11, 100, 101, 99, 102],
        }
    )
    model = Fenrir(df, variance_threshold=0.8, cluster_range=range(2, 4))
    labels = model.fit_predict()

    print("Clase:", type(model).__name__)
    print("Clusters unicos:", int(labels.nunique()))
    print("Mejor configuracion:")
    print(model.best_configuration())


if __name__ == "__main__":
    main()
