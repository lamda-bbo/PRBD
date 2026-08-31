import gzip
import os
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

class HpoSVMBench:
    """
    388-dimensional SVM benchmark used in BO.
    Self-contained, loads data from CT_slice_X.npy(.gz) and CT_slice_y.npy(.gz).
    Outputs RMSE (to minimize).
    """
    def __init__(self, data_folder=None):
        self.dims = 388

        # Load data
        self.X, self.y = self._load_data(data_folder='/data/zy/data/svm')

        # Subsample 500 points for train/test split
        np.random.seed(388)
        idxs = np.random.choice(np.arange(len(self.X)), min(500, len(self.X)), replace=False)
        half = len(idxs) // 2
        self._X_train = self.X[idxs[:half]]
        self._X_test = self.X[idxs[half:]]
        self._y_train = self.y[idxs[:half]]
        self._y_test = self.y[idxs[half:]]

        # Set bounds [0,1] for all variables (matching the original benchmark scaling)
        self.lb = np.zeros(self.dims)
        self.ub = np.ones(self.dims)

    def _load_data(self, data_folder='/data/zy/data/svm'):
        if data_folder is None:
            data_folder = os.path.join(Path(__file__).parent, "data", "svm")

        then = time.time()
        try:
            X = np.load(os.path.join(data_folder, "CT_slice_X.npy"))
            y = np.load(os.path.join(data_folder, "CT_slice_y.npy"))
        except:
            fx = gzip.GzipFile(os.path.join(data_folder, "CT_slice_X.npy.gz"), "r")
            fy = gzip.GzipFile(os.path.join(data_folder, "CT_slice_y.npy.gz"), "r")
            X = np.load(fx)
            y = np.load(fy)
            fx.close()
            fy.close()

        # scale features and target to [0,1]
        X = MinMaxScaler().fit_transform(X)
        y = MinMaxScaler().fit_transform(y.reshape(-1, 1)).squeeze()
        now = time.time()
        print(f"Loaded data in {now - then:.2f} seconds")
        return X, y

    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate the benchmark at x.
        x: np.array of shape (388,), values in [0,1]
        Returns RMSE (to minimize)
        """
        assert isinstance(x, np.ndarray)
        assert x.shape == (self.dims,)
        assert np.all(x >= self.lb) and np.all(x <= self.ub)

        # Map variables to hyperparameters for SVR
        C = 0.01 * (500 ** x[387])
        gamma = 0.1 * (30 ** x[386])
        epsilon = 0.01 * (100 ** x[385])
        length_scales = np.exp(4 * x[:385] - 2)

        # Train SVR
        svr = SVR(gamma=gamma, epsilon=epsilon, C=C, cache_size=1500, tol=0.001)
        svr.fit(self._X_train / length_scales, self._y_train)

        # Predict and compute RMSE
        pred = svr.predict(self._X_test / length_scales)
        rmse = np.sqrt(np.mean((pred - self._y_test) ** 2))
        return -rmse
