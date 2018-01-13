'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
print('x', x)

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
layer_1 = tf.nn.softmax(layer_1)

# Output layer with linear activation
out_layer = tf.matmul(layer_1, weights['out']) + biases['out']


# Parameters
learning_rate = 0.0001
training_epochs = 15
batch_size = 100

# define cost & optimizer

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

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
            #avg_cost += c / total_batch
            
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))    
            
print("Optimization Finished!")

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(out_layer, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy- Neural Network :', sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))