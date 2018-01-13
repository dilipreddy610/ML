import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

x = tf.placeholder(tf.float32, [None, 784])
print('x',x)
W1 = tf.Variable(tf.zeros([784, 512]))
b1 = tf.Variable(tf.zeros([512]))


# one hidden layer
L2 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.zeros([512, 10]))
b2 = tf.Variable(tf.zeros([10]))


y = tf.nn.relu(tf.matmul(L2, W2), b2)
print(y)

#hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)
#cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))

# To implement cross-entropy we need to first add a new placeholder
y_ = tf.placeholder(tf.float32, [None, 10])
print(y_)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# train and evaluate model
cross_entropy = tf.reduce_mean(
tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                   x: batch[0], y_: batch[1]})
            print('step %d, training accuracy %g' % (i, train_accuracy))

        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels}))