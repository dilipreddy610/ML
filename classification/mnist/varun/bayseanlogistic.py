import math
import struct
import sys

import scipy.sparse
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.label import LabelBinarizer

import matplotlib.pyplot as plt
import numpy as np


class logistic_sigmoid_model:
    def _init(self):
        self.hyper_para_bayes_logistic = hyper_para_bayes_logistic
        self.pred_bayes_logistic = pred_bayes_logistic

    def training(self, Theta, y):
        self.w0 = np.random.normal(0, 1)
        self.S0 = np.diag(np.random.normal(0, 1, y.shape[0]))
        # Theta n*m (n samples, m features), y n*1
        self.w_map, self.S_N = self.hyper_para_bayes_logistic(self.w0, self.S0, Theta, y)
    
    def predict(self, theta):
        return self.pred_bayes_logistic(self.w_map, self.S_N, theta)
    
    

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1)).T
    return sm


def getLoss(w, x, y, lam):
    m = x.shape[0]  # First we get the number of training examples
    #y_mat = oneHotIt(y) #Next we convert the integer class coding into a one-hot representation
    lb = LabelBinarizer()
    y_mat = lb.fit_transform(y)
    b = np.random.rand(len(x), 10)
    #scores = np.sum(np.dot(x, w), b)  # Then we compute raw class scores given our input and current weights
    scores = np.dot(x, w)  # Then we compute raw class scores given our input and current weights
    prob = softmax(scores)  # Next we perform a softmax on these scores to get their probabilities
    loss = (-1 / m) * np.sum(y_mat * np.log(prob)) + (lam / 2) * np.sum(w * w)  # We then find the loss of the probabilities
    grad = (-1 / m) * np.dot(x.T, (y_mat - prob)) + lam * w  # And compute the gradient for that loss
    return loss, grad

def getProbsAndPreds(someX, weights):
    probs = softmax(np.dot(someX,weights))
    preds = np.argmax(probs,axis=1)
    return probs,preds

def getAccuracy(someX,someY, weights):
    prob,prede = getProbsAndPreds(someX, weights)
    someY = someY.flatten()
    accuracy = sum(prede == someY)/(float(len(someY)))
    return accuracy



'''
Compute the stochastic gradient descent based weight vector.
 
'''
def SGD_sol(learning_rate, minibatch_size, num_epochs, L2_lambda, input_training,
            output_training):
    N, _ = input_training.shape
    # You can try different mini-batch size size
    # Using minibatch_size = N is equivalent to standard gradient descent
    # Using minibatch_size = 1 is equivalent to stochastic gradient descent
    # In this case, minibatch_size = N is better
    #weights = np.zeros([1, input_training.shape[1]])
    weights = np.random.rand(input_training.shape[1],len(np.unique(output_training)))
    #weights = np.zeros([input_training.shape[1],len(np.unique(output_training))])
        
    # We are using early stopping as a regularization technique
    for epoch in range(1, num_epochs + 1):
        losses = []
        for i in range(int(N / minibatch_size)):
            lower_bound = int(i * minibatch_size)
            upper_bound = int(min((i + 1) * minibatch_size, N))
            Phi = input_training[lower_bound : upper_bound, :]
            t = output_training[lower_bound : upper_bound]
            loss, grad = getLoss(weights, Phi, t, L2_lambda)
            losses.append(loss)
            weights = weights - (learning_rate * grad)
                            
    return weights

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

def multiclass_sigmoid_logistic(Theta, Y):
    n_class = Y.shape[1]
    models = []

    for i in range(n_class):
        models.append(logistic_sigmoid_model())
        models[i].training(Theta, Y[:, i])
    return models


def hyper_para_bayes_logistic(self, m_0, S_0, Theta, y):
    w_map = m_0
    S_N = np.linalg.inv(S_0)
    Theta = Theta.T
    for i in range(Theta.shape[0]):
        S_N = S_N + y[i]*(1-y[i])*np.matmul(Theta[i].T, Theta[i])
    return w_map, S_N

def pred_bayes_logistic(self, w_map, S_N, theta):
    mu_a = np.dot(w_map.T, theta)
    var_a = np.dot(np.dot(theta.T, S_N), theta)
    kappa_var = (1 + math.pi*var_a/8)^(-0.5)
    x = kappa_var*mu_a
    return 1/(1 + np.exp(-x))
x, y = loadmnist('../resources/mnist/train-images.idx3-ubyte', '../resources/mnist/train-labels.idx1-ubyte')

    
# starter
'''
digits = load_digits()

print(digits.data.shape)
print(digits.target.shape)
digits.target = digits.target.reshape([-1, 1])
# split in training and test data
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25)
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# t is a vector which is all 0 except 1 where we need to predict the value


patience = 10
validation_steps = 5
L2_lambdas = [0.1, 0.01, 0.03]
num_epochs = 100


weights = SGD_sol(1, int(len(x_train) / 100), num_epochs, 0.003, x_train, y_train)
#print(weights)
print(getAccuracy(x_test, y_test, weights))

