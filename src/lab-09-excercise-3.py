# MNIST deep, wide

import tensorflow as tf

tf.compat.v1.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

keep_prob = tf.placeholder(tf.float32)
nbClasses = 10
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nbClasses])

#W1 = tf.Variable(tf.random.normal([784, 256]), name='weight1')
W1 = tf.get_variable('W1', shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer_conv2d())
b1 = tf.Variable(tf.random.normal([256]), name='bias1')
_layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
layer1 = tf.nn.dropout(_layer1, keep_prob=keep_prob)

#W2 = tf.Variable(tf.random.normal([256, 256]), name='weight2')
W2 = tf.get_variable('W2', shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer_conv2d())
b2 = tf.Variable(tf.random.normal([256]), name='bias2')
_layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
layer2 = tf.nn.dropout(_layer2, keep_prob=keep_prob)

#W3 = tf.Variable(tf.random.normal([256, 256]), name='weight3')
W3 = tf.get_variable('W3', shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer_conv2d())
b3 = tf.Variable(tf.random.normal([256]), name='bias3')
_layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)
layer3 = tf.nn.dropout(_layer3, keep_prob=keep_prob)

W4 = tf.Variable(tf.random.normal([256, nbClasses]), name='weight4')
b4 = tf.Variable(tf.random.normal([nbClasses]), name='bias4')
logits = tf.matmul(layer3, W4) + b4
hypothesis = tf.nn.relu(logits)

#cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
#train = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(cost)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
train = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

isCorrect = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))

numEpochs = 15
batchSize = 100
numIterations = int(mnist.train.num_examples / (batchSize * 1e-0))

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(numEpochs):
    avgCost = 0

    for i in range(numIterations):
        bx, by = mnist.train.next_batch(batchSize)
        feed_dict={X:bx, Y:by, keep_prob: 0.7}
        _, cv = sess.run([train, cost], feed_dict=feed_dict)
        avgCost += cv / numIterations

    print("epoch: {:04d}, cost: {:.9f}".format(epoch+1, avgCost))

print("accuracy: ", sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels, keep_prob: 1}))
