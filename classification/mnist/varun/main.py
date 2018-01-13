import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image
import os


'''
This method loads the USPS data by first resizing each image to the MNIST size
an then appends it to create the X and Y values
'''
def load_usps_data(resize_width, resize_height):
    x_usps = np.ndarray(shape=(1,784),dtype=float)
    y_usps = np.ndarray(shape=(1,10),dtype=float)

    print('loading and normalizing USPS data, please wait.....')
    # iterate each number data
    for i in range(0, 10):
        y_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        y_vector[i] = 1
        
        #iterate each image
        for image_file_name in os.listdir('Numerals/' + str(i) + '/'):
            if image_file_name.endswith(".png"):
                image = Image.open('Numerals/'+str(i)+'/'+image_file_name)
                image = image.convert('1');
                image = image.resize((resize_width, resize_height), Image.ANTIALIAS)
                np.asarray(image)

                # reshape and append
                x_vector = np.reshape(image, [1, 784])
                x_vector = np.absolute(np.subtract(x_vector,1))
                x_usps = np.append(x_usps, x_vector, axis=0)
                y_vector = np.reshape(y_vector, [1, 10])
                y_usps = np.append(y_usps, y_vector, axis=0)

    print('USPS data load complete!')
    return x_usps, y_usps


# starter
usps_resize_width = 28 
usps_resize_height = 28

# get MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# get the normalized USPS test data
x_usps, y_usps = load_usps_data(usps_resize_width, usps_resize_height)

# Logistic Regression
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
# To implement cross-entropy we need to first add a new placeholder
y_ = tf.placeholder(tf.float32, [None, 10])


# cross-entropy function
cross_entropy = tf.reduce_mean( -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# apply your choice of optimization algorithm to modify the variables and reduce the loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# launch the model in an InteractiveSession
sess = tf.InteractiveSession()

# creating an operation to initialize the variables we created
tf.global_variables_initializer().run()

# running the training step 1000 times
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # check if our prediction matches the truth

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# casting to floating point numbers and then take the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# accuracy on our test data

print('Logistic Regression: Test Accuracy on MNIST test data: ', sess.run(
    accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print('Logistic Regression: Test Accuracy on USPS data: ', sess.run(
    accuracy, feed_dict={x: x_usps, y_: y_usps}))

#################################################################
#SINGLE LAYER NEURAL NETWORK
#################################################################

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
num_hidden_units = 256  # 1st hidden layer number of features

weights = {
        'h1': tf.Variable(tf.random_normal([784, num_hidden_units])),
        'out': tf.Variable(tf.random_normal([num_hidden_units, 10]))
    }

biases = {
        'b1': tf.Variable(tf.random_normal([num_hidden_units])),
        'out': tf.Variable(tf.random_normal([10]))
    }

# Hidden layer with RELU activation
layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
layer_1 = tf.nn.relu(layer_1)

# Output layer with linear activation
out_layer = tf.matmul(layer_1, weights['out']) + biases['out']


# Parameters
learning_rate = 0.003
training_epochs = 15
batch_size = 100

# define cost & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# compute metrics
cross_entropy = tf.reduce_mean(
tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out_layer))
correct_prediction = tf.equal(tf.argmax(out_layer, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Training cycle
for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/100)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / total_batch
            
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))    
            
print("Training complete!")

# Test model and check accuracy
print('Neural Network - Accuracy on MNIST test data :', sess.run(
    accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
print('Neural Network - Test Accuracy on USPS data: ', sess.run(
    accuracy, feed_dict={x: x_usps, y: y_usps}))


##########################################################################
#CONVOLUTIONAL NEURAL NETWORK
##########################################################################
'''
Compute the weight variable.
'''
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
'''
Compute the bias variable
'''
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
'''
Return a convolution implementation
'''
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# input
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# applying softmax
y = tf.nn.softmax(tf.matmul(x, W) + b)

# To implement cross-entropy we need to first add a new placeholder
y_ = tf.placeholder(tf.float32, [None, 10])

sess = tf.InteractiveSession()

# first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) +
b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# train and evaluate model
cross_entropy = tf.reduce_mean(
tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                   x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print('CNN: Test accuracy on MNIST data %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    print('CNN: Test Accuracy on USPS data: ', sess.run(
    accuracy, feed_dict={x: x_usps, y_: y_usps, keep_prob: 1.0}))
