'''
import struct
import numpy as np
import scipy.sparse
def loadmnist(imagefile, labelfile):
    # Open the images with gzip in read binary mode
    images = open(imagefile, 'rb')
    labels = open(labelfile, 'rb')

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = struct.unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = struct.unpack('>I', rows)[0]
    cols = images.read(4)
    cols = struct.unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)
    N = labels.read(4)
    N = struct.unpack('>I', N)[0]

    # Get data
    x = np.zeros((N, rows*cols), dtype=np.uint8)  # Initialize numpy array
    y = np.zeros(N, dtype=np.uint8)  # Initialize numpy array
    for i in range(N):
        for j in range(rows*cols):
            tmp_pixel = images.read(1)  # Just a single byte
            tmp_pixel = struct.unpack('>B', tmp_pixel)[0]
            x[i][j] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = struct.unpack('>B', tmp_label)[0]

    images.close()
    labels.close()
    return (x, y)


train_img, train_lbl = loadmnist('MNIST_data/train-images.idx3-ubyte', 'MNIST_data/train-labels.idx1-ubyte')
#test_img, test_lbl = loadmnist('MNIST_data/t10k-images.idx3-ubyte'
#                               , 'MNIST_data/t10k-labels.idx1-ubyte')

print(train_img.shape)
print(train_lbl.shape)

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import struct
import scipy.sparse


def loadmnist(imagefile, labelfile):
    # Open the images with gzip in read binary mode
    images = open(imagefile, 'rb')
    labels = open(labelfile, 'rb')

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = struct.unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = struct.unpack('>I', rows)[0]
    cols = images.read(4)
    cols = struct.unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)
    N = labels.read(4)
    N = struct.unpack('>I', N)[0]

    # Get data
    x = np.zeros((N, rows*cols), dtype=np.uint8)  # Initialize numpy array
    y = np.zeros(N, dtype=np.uint8)  # Initialize numpy array
    for i in range(N):
        for j in range(rows*cols):
            tmp_pixel = images.read(1)  # Just a single byte
            tmp_pixel = struct.unpack('>B', tmp_pixel)[0]
            x[i][j] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = struct.unpack('>B', tmp_label)[0]

    images.close()
    labels.close()
    return (x, y)

x, y = loadmnist('../resources/mnist/train-images.idx3-ubyte', '../resources/mnist/train-labels.idx1-ubyte')

def getLoss(w,x,y,lam):
    m = x.shape[0] #First we get the number of training examples
    y_mat = oneHotIt(y) #Next we convert the integer class coding into a one-hot representation
    scores = np.dot(x,w) #Then we compute raw class scores given our input and current weights
    prob = softmax(scores) #Next we perform a softmax on these scores to get their probabilities
    loss = (-1 / m) * np.sum(y_mat * np.log(prob)) + (lam/2)*np.sum(w*w) #We then find the loss of the probabilities
    grad = (-1 / m) * np.dot(x.T,(y_mat - prob)) + lam*w #And compute the gradient for that loss
    return loss,grad


def oneHotIt(Y):
    m = Y.shape[0]
    #Y = Y[:,0]
    OHX = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
    OHX = np.array(OHX.todense()).T
    return OHX

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

def getProbsAndPreds(someX):
    probs = softmax(np.dot(someX,w))
    preds = np.argmax(probs,axis=1)
    return probs,preds

w = np.zeros([x.shape[1],len(np.unique(y))])
lam = 1
iterations = 1000
learningRate = 1e-5
losses = []
for i in range(0,iterations):
    loss,grad = getLoss(w,x,y,lam)
    losses.append(loss)
    w = w - (learningRate * grad)
print ("loss",loss)

def getAccuracy(someX,someY):
    prob,prede = getProbsAndPreds(someX)
    accuracy = sum(prede == someY)/(float(len(someY)))
    return accuracy

print ("Accuracy",getAccuracy(x, y))

classWeightsToVisualize = 5
plt.imshow(scipy.reshape(w[:,classWeightsToVisualize],[28,28]))
plt.show()