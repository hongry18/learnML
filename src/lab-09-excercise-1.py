# MNIST deep, wide

import tensorflow as tf

tf.compat.v1.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

nbClasses = 10
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nbClasses])

W1 = tf.Variable(tf.random.normal([784, 200]), name='weight1')
b1 = tf.Variable(tf.random.normal([200]), name='bias1')
layer1 = tf.nn.softmax(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random.normal([200, 200]), name='weight2')
b2 = tf.Variable(tf.random.normal([200]), name='bias2')
layer2 = tf.nn.softmax(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random.normal([200, 200]), name='weight3')
b3 = tf.Variable(tf.random.normal([200]), name='bias3')
layer3 = tf.nn.softmax(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random.normal([200, nbClasses]), name='weight4')
b4 = tf.Variable(tf.random.normal([nbClasses]), name='bias4')
hypothesis = tf.nn.softmax(tf.matmul(layer3, W4) + b4)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

isCorrect = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))

numEpochs = 200
batchSize = 100
numIterations = int(mnist.train.num_examples / batchSize)

sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(numEpochs):
    avgCost = 0

    for i in range(numIterations):
        bx, by = mnist.train.next_batch(batchSize)
        _, cv = sess.run([train, cost], feed_dict={X:bx, Y:by})
        avgCost += cv / numIterations

    print("epoch: {:04d}, cost: {:.9f}".format(epoch+1, avgCost))

print("accuracy: ", sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
