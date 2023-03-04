import numpy as np
from random import shuffle as s

def mapSoftMaxTargets(source, value):
    def equal(item):
        return 1 if item == value else 0
    return np.array(list(map(equal, np.array(source))))

def shuffle(source):
    s(source)

    return source

def reshapeForInput(X):
    X = np.array(X)
    print(X)
    print(X.shape)
    X = X.reshape(X.shape[0], 48, 48, 1)
    X = X.astype('float32') / 255

    return X