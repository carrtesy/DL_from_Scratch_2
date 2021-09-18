import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    if x.dim == 2:
        x = x - x.max(axis = 1, keepdims = True)
        x = np.exp(x)
        x /= x.sum(axis = 1, keepdims = True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x