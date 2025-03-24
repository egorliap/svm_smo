from typing import Literal
import numpy as np
from sklearn.base import ClassifierMixin

import qpproblem_smo_solver as qp_solver


class SVMClassifier(ClassifierMixin):
    def __init__(
        self,
        kernel: Literal["linear", "polynomial", "rbf"] = "linear",
        C=1.0,
        tol=1e-3,
        max_iter=10000,
    ):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.kernel = kernel
        self.alpha = None
        self.b = 0
        self.errors = None

    def fit(self, X: np.ndarray = None, y: np.ndarray = None):
        """
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Training data features
        y : array like, shape = (n_samples)
        Training data targets
        """
        self.n_samples, n_features = X.shape
        self.alpha = np.zeros(self.n_samples)
        self.b = 0
        self.X = X
        self.y = y
        self.solver = qp_solver.QPSolver(
            list(X), list(y), self.kernel, self.C, self.tol, self.max_iter, logs=True
        )

        self.solver.solve()
        self.alpha = self.solver.get_alpha()
        self.b = self.solver.get_b()

    def objective(self, X: np.ndarray):
        ans = self.solver.output(X)
        result = np.sign(ans)
        result[result == 0] = -1
        return result

    def decision_function(self, X):
        return np.array(self.solver.output(X))

    def predict(self, X=None):
        """
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Data to predict
        """

        return self.objective(X)
