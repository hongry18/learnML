import tensorflow as tf
import random

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 8
#config.inter_op_parallelism_threads = 8

from tensorflow.examples.tutorials.mnist import input_data

tf.compat.v1.set_random_seed(777)

mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)

learningRate = 1e-3
trainingEpochs = 15
batchSize = 100
totalBatch = int(mnist.train.num_examples / batchSize)

keep_prob = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, 784])
XImg = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random.normal([3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(XImg, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.Variable(tf.random.normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.Variable(tf.random.normal([3, 3, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3f = tf.reshape(L3, [-1, 128 * 4 * 4])

W4 = tf.get_variable('W4', shape=[128*4*4, 625], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random.normal([625]))
L4 = tf.nn.relu(tf.matmul(L3f, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable('W5', shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random.normal([10]))
logits = tf.matmul(L4, W5) + b5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
train = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

sess = tf.Session(config=config)
sess.run(tf.compat.v1.global_variables_initializer())

print('learning start')
for epoch in range(trainingEpochs):
    avgCost = 0

    for i in range(totalBatch):
        bx, by = mnist.train.next_batch(batchSize)
        fd = {X: bx, Y: by, keep_prob: 0.7}
        _, c = sess.run([train, cost], feed_dict=fd)
        avgCost += c / totalBatch

    #print('Epoch: %04d, cost: %.9f' % ( (epoch+1), avgCost ) )
    print('Epoch: {:04d}, cost: {:.9f}'.format((epoch+1), avgCost) )
print('learning finish')

correctPrediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

print('accuracy: {:.9f}'.format(sess.run(accuracy, feed_dict={X: mnist.test.images, Y:mnist.test.labels, keep_prob: 1.})))

def evaluate(x, y, size=512):
    N = x.shape[0]
    correctSample = 0

    for i in range(size):
        bx = x[i: i+size]
        by = y[i: i+size]
        bN = bx.shape[0]

        feed = {X: bx, Y: by, keep_prob: 1}

        correctSample += sess.run(accuracy, feed_dict=feed) * bN

    return correctSample / N

print("\nAccuracy Evaluates")
print("-------------------------------")
print('Train Accuracy:', evaluate(mnist.train.images, mnist.train.labels))
print('Test Accuracy:', evaluate(mnist.test.images, mnist.test.labels))


# Get one and predict
print("\nGet one and predict")
print("-------------------------------")
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(tf.argmax(logits, 1), {X: mnist.test.images[r:r + 1], keep_prob: 1}))
