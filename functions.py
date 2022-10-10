'''Numpy implementation of some cost functions, activation functions, their derivatives and other.'''
import numpy as np
from Function import Function, Quadratic
import matplotlib.pyplot as plt

class Sigmoid(Function):
    def __init__(self):
        pass

    def __call__(self, X):
        return 1/(1+np.exp(-X))

    def get_derivative(self, X):
        return self(X) * (1 - self(X))

class Relu(Function):
    def __init__(self):
        pass

    def __call__(self, X):
        return np.maximum(0,X)

    def get_derivative(self, X: np.ndarray) -> np.ndarray:
        dX = np.full_like(X, 1)
        dX[X < 0] = 0
        return dX

class CrossEntropy(Function):
    '''Cross entropy error function  of predictions Y_ and reference Y.'''
    def __init__(self):
        pass

    def __call__(self, Y_, Y):
        logprobs = np.multiply(Y, np.log(Y_)) + np.multiply((1-Y), np.log(1-Y_))
        J = -np.mean(logprobs)
        return np.squeeze(J)

    def get_derivative(self, Y_, Y):
        # TODO test 
        return -(np.divide(Y, Y_) - np.divide(1 - Y, 1 - Y_))

if __name__ == "__main__":
    sigmoid = Sigmoid()
    relu = Relu()
    print(sigmoid(1))
    print(sigmoid.get_derivative(1))
    xs = np.array(list(range(-100,100)))/10
    fig, axes = plt.subplots(2,2)

    axes[0,0].plot(xs,sigmoid(xs))
    axes[0,0].grid()
    axes[0,0].set_title("Sigmoid")

    axes[0,1].plot(xs,relu(xs))
    axes[0,1].grid()
    axes[0,1].set_title("Relu")

    axes[1,0].plot(xs, sigmoid.get_derivative(xs))
    axes[1,0].grid()
    axes[1,0].set_title("Sigmoid derivative")

    axes[1,1].plot(xs, relu.get_derivative(xs))
    axes[1,1].grid()
    axes[1,1].set_title("Relu derivative")

    plt.tight_layout()
    plt.show()