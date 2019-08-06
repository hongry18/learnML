# Lab 11 MNIST and Convolutional Neural Network
import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data

tf.compat.v1.set_random_seed(777)

mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)

learningRate = 1e-3
traningEpochs = 15
batchSize = 100
totalBatch = int(mnist.train.num_examples / batchSize)

X = tf.placeholder(tf.float32, [None, 784])
xImg = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape = (?, 28, 28, 1)
W1 = tf.Variable(tf.random.normal([3,3,1,32], stddev=0.01))
# Conv -> (?, 28, 28, 32)
# Pool -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(xImg, W1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
'''
Tensor("Conv2D: 0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu: 0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool: 0", shape=(?, 14, 14, 32), dtype=float32)
'''

# L2 ImgIn shape = (?, 14, 14, 32)
W2 = tf.Variable(tf.random.normal([3,3,32,64], stddev=0.01))
# Conv -> (?, 14, 14, 64)
# Pool -> (?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L2Flat = tf.reshape(L2, [-1, 7*7*64])
'''
Tensor("Conv2D_1: 0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("Relu_1: 0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("MaxPool_1: 0", shape=(?, 7, 7, 64), dtype=float32)
Tensor("Reshape_1: 0", shape=(?, 3136), dtype=float32)
'''

W3 = tf.get_variable('W3', shape=[7*7*64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random.normal([10]))
logits = tf.matmul(L2Flat, W3) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer())

print('Learning started. It takes sometime.')
for epoch in range(traningEpochs):
    avgCost = 0

    for i in range(totalBatch):
        bx, by = mnist.train.next_batch(batchSize)
        fd = {X: bx, Y: by}
        _, c = sess.run([optimizer, cost], feed_dict=fd)
        avgCost += c / totalBatch

    print('Epoch: ', '%04d' % (epoch + 1), ', cost :', '%.9f' % (avgCost))

print('Learning finished')

correctPrediction = tf.equal(tf.argmax(logits,1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
print('accuracy: ', sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))

r = random.randint(0, mnist.test.num_examples-1)
print('label: ', sess.run(tf.argmax(mnist.test.labels[r: r+1], 1)))
print('prediction: ', sess.run(tf.argmax(logits,1), feed_dict={X: mnist.test.images[r: r+1]}))
