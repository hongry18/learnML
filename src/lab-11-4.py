# Lab 11 - 4 using layers of tensorflow
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

tf.compat.v1.set_random_seed(777)

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 8

mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)

learningRate = 1e-3
trainingEpochs = 15
batchSize = 100
totalBatch = int(mnist.train.num_examples / batchSize)

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.training = tf.placeholder(tf.bool)
            self.X = tf.placeholder(tf.float32, [None, 784])
            XImg = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            # Convolutional Layer #1
            c1 = tf.layers.conv2d(inputs=XImg, filters=32, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu)
            # Pooling Layer #1
            p1 = tf.layers.max_pooling2d(inputs=c1, pool_size=[2, 2], padding='SAME', strides=2)
            # dropout
            d1 = tf.layers.dropout(inputs=p1, rate=0.3, training=self.training)

            # Convolutional Layer #2
            c2 = tf.layers.conv2d(inputs=d1, filters=64, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu)
            # Pooling Layer #2
            p2 = tf.layers.max_pooling2d(inputs=c2, pool_size=[2, 2], padding='SAME', strides=2)
            # dropout
            d2 = tf.layers.dropout(inputs=p2, rate=0.3, training=self.training)

            # Convolutional Layer #3
            c3 = tf.layers.conv2d(inputs=d2, filters=128, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu)
            # Pooling Layer #3
            p3 = tf.layers.max_pooling2d(inputs=c3, pool_size=[2, 2], padding='SAME', strides=2)
            # dropout
            d3 = tf.layers.dropout(inputs=p3, rate=0.3, training=self.training)

            # Dense Layer with Relu
            flat = tf.reshape(d3, [-1, 128 * 4 * 4])
            dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
            d4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.training)

            # Logits
            self.logits = tf.layers.dense(inputs=d4, units=10)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(self.cost)

        correctPrediction = tf.equal( tf.argmax(self.logits, 1), tf.argmax(self.Y, 1) )
        self.accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

    def predict(self, x, training=False):
        return self.sess.run(self.logits, feed_dict={self.X: x, self.training: training})

    def getAccuracy(self, x, y, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x, self.Y: y, self.training: training})

    def train(self, x, y, training=False):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x, self.Y: y, self.training: training})

sess = tf.Session(config=config)
m1 = Model(sess, 'm1')

sess.run(tf.compat.v1.global_variables_initializer())

print('learining started')

for epoch in range(trainingEpochs):
    avgCost = 0

    for i in range(totalBatch):
        bx, by = mnist.train.next_batch(batchSize)
        c, _ = m1.train(bx, by)
        avgCost += c / totalBatch

    print('Epoch: {:04d}, cost: {:.9f}'.format( (epoch+1), avgCost))

print('learining finished')

print('accuracy: ', m1.getAccuracy(mnist.test.images, mnist.test.labels))
