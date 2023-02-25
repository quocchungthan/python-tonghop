import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np

def loadSingleImageByPath(path):
    image = mpimg.imread(path)
    plt.imshow(image)
    plt.show()

def loadSingleImageAsNumber(path):
    image = mpimg.imread(path)
    return np.array(image)

def allFileNames(folder):
    return os.listdir(folder)