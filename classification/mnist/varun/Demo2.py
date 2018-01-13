from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from _struct import unpack
from numba.tests.npyufunc.test_ufunc import dtype


def load_mnist(imagefile, labelfile):
    images = open(imagefile, 'rb')
    labels = open(labelfile, 'rb')
    
    # get metadata for images
    images.read(4) # skip magic number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # get metadata for labels
    labels.read(4)
    N = labels.read(4)
    N = unpack('>I', N)[0]
    
    # get data
    x = np.zeros((N, rows*cols), dtype=np.uint8) # initialize
    y = np.zeros(N, dtype=np.uint8)
    
    for i in range(N):
        for j in range(rows*cols):
            tmp_pixel = images.read(1) # just a single byte
            tmp_pixel = unpack('>B', tmp_pixel)[0] 
            x[i][j] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0] 
    
    images.close()
    labels.close()
    return (x, y)           
    

# starter
train_img, train_lbl = load_mnist('../resources//mnist//train-images.idx3-ubyte', 
                                  '../resources//mnist//train-labels.idx1-ubyte')

test_img, test_lbl = load_mnist('../resources//mnist//t10k-images.idx3-ubyte', 
                                  '../resources//mnist//t10k-labels.idx1-ubyte')

print(train_img.shape)
print(train_lbl.shape)
print(test_img.shape)
print(test_lbl.shape)


# show images with labels
plt.figure(figsize=(20, 4))
for index, (image, label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.get_cmap('gray'))
    plt.title('Training: %i\n' % label, fontsize=20)   
plt.show()




