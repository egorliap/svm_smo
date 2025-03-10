import numpy as np

from scr.base_kernel import Kernel


class NH:
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def __contains__(self, num: float):
        if num > self.a and num < self.b:
            return True
        return False


class SMO_QPSolver:
    """
    This class represents a Sequential Minimal Optimization (SMO) algorithm for solving the quadratic
    programming problem in the context of support vector machines.
    Solves the SVM optimization problem (the support vector method) in a dual form.

    Target function:
            min_α Ψ(α) = 0.5 * ΣΣ y_i y_j K(x_i, x_j) α_i α_j - Σ α_i
    Under restrictions:
            1. 0 ≤ α_i ≤ C, ∀i
            2. Σ y_i α_i = 0

        where:
    - K(x_i, x_j) is the pos-def kernel function

    - C is the regularization parameter

    - α_i are Lagrange multipliers

    - y_i ∈ {-1, +1} are class labels
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        K: Kernel = lambda x, y: np.dot(x, y),
        C: float = 1.0,
        tol: float = 1e-3,
        max_iter: int = 10000,
    ):
        self.X = X
        self.y = y
        self.K = K
        self.C = C
        self.tol = tol
        self.max_iter = max_iter

        self.n_samples = X.shape[0]
        if self.n_samples != y.size:
            raise Exception(
                f"y must be the same length as X, but {X.shape=} and {y.size=}"
            )

        self.errors = np.array([-10000000 for _ in range(self.n_samples)])

        print(self.X, self.y)

    def solve(self, alpha: np.ndarray, b: float):
        if alpha.size != self.n_samples:
            raise Exception(
                f"alpha must be the same length as X, but {self.X.shape=} and {alpha.size=}"
            )
        iteration = 0
        while iteration < self.max_iter:
            prev_js = []
            prev_is = []
            # 1. Выбор первого множителя (нарушителя KKT)
            i = self.select_first_multiplier(alpha, b)
            if i is None:
                break  # Все примеры удовлетворяют KKT
            if self.errors[i] != -10000000:
                E1 = self.errors[i]
            else: 
                E1 = self.compute_error(self.X[i], self.y[i], alpha, b)
            r = E1 * self.y[i]
            if  not ((r < -self.tol and alpha[i] < self.C) or (r > self.tol and alpha[i] > 0)):
                continue
            # 2. Выбор второго множителя
            j = self.select_second_multiplier(i, alpha, b)

            if j is None:
                continue

            # 3. Совместная оптимизация alpha[i] и alpha[j]
            while True:
                prev_js.append(j)

                j = self.select_second_multiplier(i, alpha, b, prev_js)
                if j is None:
                    break
                if self.take_step(i, j, alpha, b):
                    break

            # 4. Обновление кэша ошибок
            self.errors[i] = self.compute_error(self.X[i], self.y[i], alpha, b)
            self.errors[j] = self.compute_error(self.X[j], self.y[j], alpha, b)

            print(iteration)
            iteration += 1

    def select_first_multiplier(self, alpha, b, prev_is=None):
        non_bound = np.where(
            ((alpha > self.tol) | (alpha > 0))
            & ((alpha < self.C - self.tol) | (alpha < self.C))
        )[0]
        for i in non_bound:
            if self.violates_KKT(self.X[i], self.y[i], alpha[i], alpha, b):
                return i
        left = np.where(
            ((alpha <= self.tol) & (alpha >= -self.tol))
            & ((alpha >= self.C - self.tol) | (alpha <= self.C + self.tol))
        )[0]

        for i in left:
            if self.violates_KKT(self.X[i], self.y[i], alpha[i], alpha, b):
                return i
        return None

    def select_second_multiplier(self, i, alpha, b, prev_js=None):
        max_delta = 0
        j = -1
        candidates = []
        if self.errors[i] != -10000000:
            Ei = self.errors[i]
        else:
            Ei = self.compute_error(self.X[i], self.y[i], alpha, b)
        if np.min(self.errors) != np.max(self.errors):
            if Ei >= 0:
                candidates = np.where(self.errors == np.min(self.errors))[0]
            else:
                candidates = np.where(self.errors == np.max(self.errors))[0]

        for candidate in candidates:
            if candidate == i or (prev_js is not None and candidate in prev_js):
                continue
            delta = abs(Ei - self.errors[candidate])
            if delta > max_delta:
                max_delta = delta
                j = candidate

        if j != -1:
            return j

        non_bound = np.where(
            ((alpha > self.tol) | (alpha > 0))
            & ((alpha < self.C - self.tol) | (alpha < self.C))
        )[0]
        
        random_indices = np.random.permutation(non_bound)
        for idx in random_indices:
            if idx == i or (prev_js is not None and idx in prev_js):
                continue
            return idx
        
        random_indices = np.random.permutation(self.n_samples)
        for idx in random_indices:
            if idx == i or (prev_js is not None and idx in prev_js):
                continue
            return idx
        return None

    def take_step(self, i1, i2, alpha, b):
        if i1 == i2:
            return 0
        alph1 = alpha[i1]
        y1 = self.y[i1]
        alph2 = alpha[i2]
        y2 = self.y[i2]
        if self.errors[i1] != -10000000:
            E1 = self.errors[i1]
        else:
            E1 = self.compute_error(self.X[i1], y1, alpha, b)

        if self.errors[i2] != -10000000:
            E2 = self.errors[i2]
        else:
            E2 = self.compute_error(self.X[i2], y2, alpha, b)

        s = y1 * y2
        L = max(0, alph2 - alph1) if y1 != y2 else max(0, alph2 + alph1 - self.C)
        H = (
            min(self.C, self.C + alph2 - alph1)
            if y1 != y2
            else min(self.C, alph2 + alph1)
        )

        if L == H:
            return 0

        k11 = self.K(self.X[i1], self.X[i1])
        k12 = self.K(self.X[i1], self.X[i2])
        k22 = self.K(self.X[i2], self.X[i2])
        eta = k11 + k22 - 2 * k12
        if eta > 0:
            a2 = alph2 + y2 * (E1 - E2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H

        else:
            Lobj = self.output(L, alpha, b)
            Hobj = self.output(H, alpha, b)
            if Lobj < Hobj - self.tol:
                a2 = L
            elif Lobj > Hobj + self.tol:
                a2 = H
            else:
                a2 = alph2

        if abs(a2 - alph2) < self.tol * (a2 + alph2 + self.tol):
            return 0
        a1 = alph1 + s * (alph2 - a2)

        b1 = E1 + y1 * (a1 - alpha[i1]) * k11 + y2 * (a2 - alpha[i2]) * k12 + b
        b2 = E2 + y1 * (a1 - alpha[i1]) * k12 + y2 * (a2 - alpha[i2]) * k22 + b
        b = (b1 + b2) / 2
        alpha[i2] = a2
        alpha[i1] = a1
        print(i1, i2)

        return 1

    def output(self, x_sample: np.ndarray, alpha, b):
        ans = 0
        for i in range(self.n_samples):
            ans += self.y[i] * alpha[i] * self.K(x_sample, self.X[i])
        return ans - b

    def update_errors(self, alpha, b):
        for i in range(self.n_samples):
            self.errors[i] = self.compute_error(self.X[i], self.y[i], alpha, b)

    def compute_error(self, x, y, alpha, b):
        return self.output(x, alpha, b) - y

    def violates_KKT(self, x, y, alpha_, alpha, b):
        y_pred = self.output(x, alpha, b)
        if (alpha_ in NH(self.C - self.tol, self.C + self.tol)) and (
            y * y_pred > 1 + self.tol
        ):
            return True
        if (alpha_ in NH(-self.tol, self.tol)) and (y * y_pred < 1 - self.tol):
            return True
        return False
