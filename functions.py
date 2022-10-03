'''Numpy implementation of some cost functions, activation functions, their derivatives and other.'''

import numpy as np

def sigmoid(X: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-X))

def sigmoid_derivative(X: np.ndarray) -> np.ndarray:
    return sigmoid(X)*(1-sigmoid(X))

def relu(X: np.ndarray) -> np.ndarray:
    return np.maximum(0,X)

def relu_derivative(X: np.ndarray) -> np.ndarray:
    dX = np.full_like(X, 1)
    dX[X < 0] = 0
    return dX


if __name__ == "__main__":
    pass 
    #TODO test functions
