import os

import numpy as np
import tensorflow as tf


# Make a queue of file names including all the JPEG images files in the relative
# image directory.
for i in range(0, 10):
    filename_queue = tf.train.string_input_producer(
       tf.train.match_filenames_once('C:/Users/rathj/Downloads/img_align_celeba/*.jpg'))


    # Read an entire image file which is required since they're JPEGs, if the images
    # are too large they could be split in advance to smaller files or use the Fixed
    # reader to split up the file.
    image_reader = tf.WholeFileReader()

    # Read a whole file from the queue, the first returned value in the tuple is the
    # filename which we are ignoring.
    _, image_file = image_reader.read(filename_queue)


    image_orig = tf.image.decode_jpeg(image_file, channels=3)
    image = tf.image.resize_images(image_orig, [28, 28])
    image.set_shape((28, 28, 3))
    batch_size = 2000
    num_preprocess_threads = 1
    min_queue_examples = 256

    images = tf.train.shuffle_batch(
    [image],
    batch_size=batch_size,
    num_threads=num_preprocess_threads,
    capacity=min_queue_examples + 3 * batch_size,
    min_after_dequeue=min_queue_examples)

# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.

# Start a new session to show example output.
with tf.Session() as sess:
    # launch the model in an InteractiveSession
    # creating an operation to initialize the variables we created
    tf.local_variables_initializer().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    # Get an image tensor and print its value.
    image_tensor = sess.run(images)
    #print(image_tensor)
       
    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
 
print(image_tensor)
print(tf.shape(image_tensor))

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
        #batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                   x: images, y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print('CNN: Test accuracy on MNIST data %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    print('CNN: Test Accuracy on USPS data: ', sess.run(
    accuracy, feed_dict={x: x_usps, y_: y_usps, keep_prob: 1.0}))
