import numpy as np
from sklearn.metrics import accuracy_score
from typing import Tuple, Callable

def make_knn_predictions(n_neighbors:int, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray) -> Tuple[Callable, float]:
    try:
        import cudf, cuml
        from cuml.neighbors import KNeighborsClassifier
    except ImportError:
        from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    return knn, acc
