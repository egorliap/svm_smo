import numpy as np


class Kernel:

    def __init__(self, kernel_function):
        if self._check_if_mercel(kernel_function):
            self.func = kernel_function
        else:
            raise Exception("Kernel function is not positive definite!")

    def __call__(self, x1: np.ndarray, x2: np.ndarray):
        return self.func(x1, x2)

    def _check_if_mercel(kernel_function):
        return True
