import numpy as np
import matplotlib.pyplot as plt
import abc

class Function(abc.ABC):
    @abc.abstractmethod # TOOD should it be abstract?
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def __call__(self, *args):
        pass

    @abc.abstractmethod
    def get_derivative(self, *args):
        pass

class Quadratic(Function):
    def __init__(self, A = 1, B = 0, C = 0):
        if A == 0:
            raise ValueError(f"Use A != 0.")
        self.A = A
        self.B = B
        self.C = C

    def __call__(self, X):
        return self.A*X**2 + self.B*X + self.C

    def get_derivative(self, X):
        return 2*self.A*X + self.B

if __name__ == "__main__":
    f1 = Quadratic()
    assert f1(0) == 0
    assert f1(1) == 1
    assert f1(2) == 4
    assert f1.get_derivative(0) == 0
    assert f1.get_derivative(1) == 2
    assert f1.get_derivative(2) == 4



