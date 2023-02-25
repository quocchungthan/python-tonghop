import numpy as np
from random import shuffle as s

def mapSoftMaxTargets(source, value):
    def equal(item):
        return 1 if item == value else 0
    return np.array(list(map(equal, np.array(source))))

def shuffle(source):
    s(source)

    return source