import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def loadSingleImageByPath(path):
    image = mpimg.imread(path)
    plt.imshow(image)
    plt.show()

def allFileNames(folder):
    return os.listdir(folder)