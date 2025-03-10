import numpy as np
from sklearn.base import ClassifierMixin

from scr.base_kernel import Kernel
from scr.qp_solver import SMO_QPSolver


class SVMClassifier(ClassifierMixin):
    def __init__(self, K: Kernel = lambda x, y: np.dot(x, y), C=1.0, tol=1e-3, max_iter=10000):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.K = K
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
        self.errors = np.zeros(self.n_samples) - 1
        self.X = X
        self.y = y
        solver = SMO_QPSolver(X, y, self.K, self.C, self.tol, self.max_iter)
        
        solver.solve(self.alpha, self.b)
    
    def objective(self, X: np.ndarray):
        ans = 0
        for i in range(self.n_samples):
            ans += self.y[i] * self.alpha[i] * self.K(X, self.X[i])
        ans -= self.b
        result = np.sign(ans)
        result[result == 0] = -1 
        return result
    
    def support(self):
        ans = 0
        for i in range(self.n_samples):
            ans += self.y[i] * self.alpha[i] * self.X[i]
        return ans
    def predict(self, X=None):
        """
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Data to predict
        """
        return self.objective(X)
