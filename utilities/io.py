import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def loadSingleImageByPath(path):
    image = mpimg.imread(path)
    plt.imshow(image)
    plt.show()
