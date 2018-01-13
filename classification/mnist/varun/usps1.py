import numpy as np
from PIL import Image
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.pyplot import contourf
from numpy import arange
from librosa.display import cmap


def usps_test_data():
    x_usps = np.ndarray(shape=(1,784),dtype=float)
    y_usps = np.ndarray(shape=(1,10),dtype=float)

    for i in range(0, 10):
        y_t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        y_t[i] = 1
        for image_file_name in os.listdir('../USPS/Numerals/' + str(i) + '/'):
            if image_file_name.endswith(".png"):
                im = mpimg.imread('USPS_norm_data/Numerals/' + str(i) + '/' + image_file_name)
                x_t = np.reshape(im, [1, 784])
                x_t = np.absolute(np.subtract(x_t,1))
                x_usps = np.append(x_usps, x_t, axis=0)
                y_t = np.reshape(y_t, [1, 10])
                y_usps = np.append(y_usps, y_t, axis=0)

    return x_usps, y_usps

#x_u, y_u = usps_test_data()
contourf(arange(-5, 5, 0.1), arange(-5, 5, 0.1), [[5, 5, 5], [5, 5, 5], [5,5,5]], cmap='Paired', levels=(0, 4, 0.1))
plt.show()



