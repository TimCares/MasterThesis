import cudf, cuml
from cuml.neighbors import KNeighborsClassifier as cuKNeighbors
from config import DATA_PATH
import os
import pickle
import numpy as np

def get_knn_model(postfix:str=None):
    if postfix is None:
        postfix = ""
    else:
        postfix = "_" + postfix
    path = os.path.join(DATA_PATH, f"knn_model{postfix}.pkl")
    if not os.path.exists(path):
        make_knn_model
    with open(path, "rb") as f:
        knn = pickle.load(f)
    return knn

def make_knn_predictions(data:np.ndarray):
    knn = get_knn_model()
    return knn.predict(data)

def make_knn_model(n_neighbors:int, X_train:np.ndarray, y_train:np.ndarray, postfix:str=None) -> None:
    model = cuKNeighbors(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    if postfix is None:
        postfix = ""
    else:
        postfix = "_" + postfix
    with open(os.path.join(DATA_PATH, f"knn_model{postfix}.pkl"), "wb") as f:
        pickle.dump(model, f)
