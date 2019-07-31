# Lab 07 Learning rate and Evaluation with mnist

import tensorflow as tf
import random
import matplotlib.pyplot as plt

tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets("./MNIST_DATA/", ont_hot=True)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nbClasses = 10

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nbClasses])

W = tf.Variable(tf.random_normal([784, nbClasses]), name='weight')
b = tf.Variable(tf.random_normal([nbClasses]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost = tf.reduce_mean( -tf.reduce_sum(Y * tf.log(hypothesis), axis=1) )
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# test Model
isCorrect = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))

numEpochs = 15
batchSize = 100
numIterations = int(mnist.train.num_examples / batchSize)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(numEpochs):
    avgCost = 0

    for i in range(numIterations):
        batchX, batchY = mnist.train.next_batch(batchSize)
        _, costVal = sess.run([train, cost], feed_dict={X: batchX, Y: batchY})
        avgCost += costVal / numIterations

    print("epoch: {:04d}, cost{:.9f}".format(epoch+1, avgCost))

# test the model using test set
print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))

r = random.randint(0, mnist.test.num_examples - 1)
print("label: ", sess.run(tf.argmax(mnist.test.labels[r : r+1], 1)))
print("prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X:mnist.test.images[r: r+1]}))

plt.imshow(mnist.test.images[r: r+1].reshape(28, 28), cmap="Greys", interpolation="nearest")
plt.show()
