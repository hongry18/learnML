# MNIST deep, wide

import tensorflow as tf

tf.compat.v1.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

nbClasses = 10
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nbClasses])

with tf.name_scope('Layer1'):
    W1 = tf.Variable(tf.random.normal([784, 40]), name='weight1')
    b1 = tf.Variable(tf.random.normal([40]), name='bias1')
    layer1 = tf.nn.softmax(tf.matmul(X, W1) + b1)

    tf.summary.histogram("W1", W1)
    tf.summary.histogram("b1", b1)
    tf.summary.histogram("Layer1", layer1)

with tf.name_scope('Layer2'):
    W2 = tf.Variable(tf.random.normal([40, 40]), name='weight2')
    b2 = tf.Variable(tf.random.normal([40]), name='bias2')
    layer2 = tf.nn.softmax(tf.matmul(layer1, W2) + b2)

    tf.summary.histogram("W2", W2)
    tf.summary.histogram("b2", b2)
    tf.summary.histogram("Layer2", layer2)

with tf.name_scope('Layer3'):
    W3 = tf.Variable(tf.random.normal([40, 40]), name='weight3')
    b3 = tf.Variable(tf.random.normal([40]), name='bias3')
    layer3 = tf.nn.softmax(tf.matmul(layer2, W3) + b3)

    tf.summary.histogram("W3", W3)
    tf.summary.histogram("b3", b3)
    tf.summary.histogram("Layer3", layer3)

with tf.name_scope('Layer4'):
    W4 = tf.Variable(tf.random.normal([40, nbClasses]), name='weight4')
    b4 = tf.Variable(tf.random.normal([nbClasses]), name='bias4')
    hypothesis = tf.nn.softmax(tf.matmul(layer3, W4) + b4)

    tf.summary.histogram("W4", W4)
    tf.summary.histogram("b4", b4)
    tf.summary.histogram("Layer4", hypothesis)

with tf.name_scope('Cost'):
    cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
    tf.summary.scalar('Cost', cost)

with tf.name_scope('Train'):
    train = tf.train.GradientDescentOptimizer(learning_rate=0.8).minimize(cost)

isCorrect = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))
tf.summary.scalar("accuracy", accuracy)

numEpochs = 30
batchSize = 100
numIterations = int(mnist.train.num_examples / batchSize)

sess = tf.Session()

mergedSummary = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs/lab-09-mnist_r0_08')
writer.add_graph(sess.graph)

sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(numEpochs):
    avgCost = 0

    for i in range(numIterations):
        bx, by = mnist.train.next_batch(batchSize)
        _, cv, summary = sess.run([train, cost, mergedSummary], feed_dict={X:bx, Y:by})
        avgCost += cv / numIterations
        writer.add_summary(summary, global_step=(epoch+1)*i+1)

    print("epoch: {:04d}, cost: {:.9f}".format(epoch, avgCost))

print("accuracy: ", sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
