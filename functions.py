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

def cross_entropy(Y_: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Computes the cross-entropy cost between Y_ and Y vectors. Returns 0-dimensional np.ndarray (shape == ()).
    
    Args:
        Y_: array of size (k,m)
        Y:  array of size (k,m)
    """
    # TODO What if k != 1?
    if Y_.shape != Y.shape:
        raise ValueError(f"Shapes of Y_ {Y_.shape} and Y {Y.shape} should be the same.")
    if Y.shape[0] != 1:
        raise NotImplementedError(f"Not implemented for k!=1.")
    logprobs = np.multiply(Y, np.log(Y_)) + np.multiply((1-Y), np.log(1-Y_))
    J = -np.mean(logprobs)
    return np.squeeze(J)

if __name__ == "__main__":
    pass 
    #TODO test functions
