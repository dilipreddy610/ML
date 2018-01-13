import os

from skimage import io
from skimage import transform
from skimage.io._io import imread
from PIL import Image
import numpy as np
import tensorflow as tf


'''
Method to load the image data
@param img_dir: directory from where images can be fetched
@param num_imgs: number of images to be processed
@param width: width of the resized image
@param height: height of the resized image

returns: numpy array if the image data    
'''
def load_img_data(img_dir, num_imgs, width, height, num_channels):
    list_of_imgs = []
    count = 0
    for img in os.listdir(img_dir):
        if not img.endswith(".jpg"):
                continue
        img = img_dir + img
        #img_arr = transform.resize(imread(img), (56, 56), mode='symmetric')
        #img_arr = np.reshape(img_arr, (56, 56, 3))
        
        # using PIL so allow AntiAliasing while resizing
        img_arr = Image.open(img)
        img_arr=img_arr.resize((width, height), Image.ANTIALIAS)
        img_arr = np.reshape(img_arr, (width, height, num_channels))

        if img_arr is None:
            print("Unable to read image", img)
            continue
        list_of_imgs.append(img_arr)
        count += 1
        
        # exit if num of images reached
        if(count == num_imgs):
            break
        
    print("Data Size: ", np.array(list_of_imgs).shape)
    return np.array(list_of_imgs)

'''
Method to load the label data
@param img_label_path: absolute path of the labels file
@param num_labels: number of labels to be fetched  
'''
def load_label(img_label_path, num_labels):
    data = np.loadtxt(img_label_path, dtype=bytes,usecols=(0,16)).astype(str)
    data = np.delete(data, (0), axis=0)
    y_celebA = np.ndarray(shape=(1,2),dtype=float)
    y_celebA = np.delete(y_celebA, (0), axis=0) 
    
    for i in range (data.shape[0]):
        # if no glass is worn
        if ( int(data[i][1]) is -1):
            y_t = [1, 0]
        else:
            # when wearing glasses 
            y_t = [0, 1]
            
        y_t = np.reshape(y_t, [1, 2])
        y_celebA = np.append(y_celebA, y_t, axis=0)    
    
        # exit if num of labels reached
        if(i + 1 == num_labels):
            break
        
    return y_celebA    

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

# image parameters
width = 56
height = 56
num_channels = 3

# data 
img_dir = 'img_align_celeba/'
label_dir = 'list_attr_celeba.txt'
num_imgs = 100000

# load data
data = load_img_data(img_dir, num_imgs, width, height, num_channels)
label_data = load_label(label_dir, num_imgs)

# split data
train_test_split_percent = 0.8
train_data = data[0:int(train_test_split_percent * num_imgs)]
train_label_data = label_data[0: int(train_test_split_percent * num_imgs)]

test_data = data[int(train_test_split_percent * num_imgs):]
test_label_data = label_data[int(train_test_split_percent * num_imgs):]

# input
x = tf.placeholder(tf.float32, [None, width, height, num_channels])

# To implement cross-entropy we need to first add a new placeholder
y_ = tf.placeholder(tf.float32, [None, 2])

sess = tf.InteractiveSession()

# first convolutional layer
W_conv1 = weight_variable([5, 5, num_channels, 64])
b_conv1 = bias_variable([64])
x_image = tf.reshape(x, [-1, width, height, num_channels])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
W_conv2 = weight_variable([5, 5, 64, 128])
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) +
b_conv2)

# third convolutional layer
W_conv3 = weight_variable([5, 5, 128, 256])
b_conv3 = bias_variable([256])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) +
b_conv3)

# second pooling layer
h_pool2 = max_pool_2x2(h_conv3)

# densely connected layer
W_fc1 = weight_variable([int(width / 4) * int(height / 4) * 256, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, int(width/4) * int(height/4)*256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# train and evaluate model
cross_entropy = tf.reduce_mean(
tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1):
        BATCH_SIZE = 500
        for start in range(0, len(train_label_data), BATCH_SIZE):
            batch_x = train_data[start: start + BATCH_SIZE]
            batch_y = train_label_data[start: start + BATCH_SIZE]
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                   x: batch_x, y_: batch_y, keep_prob: 1.0})
                print('step %d, training accuracy %g' % (start, train_accuracy))

            train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
    print('CNN: Test accuracy on CelebA data %g' % accuracy.eval(feed_dict={
        x: test_data, y_: test_label_data, keep_prob: 1.0}))
   