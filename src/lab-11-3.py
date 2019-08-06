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

    def _getConvLayer(self, x, f, i, o, isFlat=False, flatSize=0):
        # X: input, f: filter, i: channels, o: outputCnt
        W = tf.Variable(tf.random.normal([f, f, i, o], stddev=0.01))
        L = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        L = tf.nn.relu(L)
        L = tf.nn.max_pool(L, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        L = tf.nn.dropout(L, keep_prob=self.keep_prob)
        if not isFlat:
            return L

        return tf.reshape(L, [-1, flatSize])

    def _getFcLayer(self, name, x, i, o):
        W = tf.get_variable(name, shape=[i, o], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.random.normal([o]))
        L = tf.nn.relu(tf.matmul(x, W) + b)
        return tf.nn.dropout(L, keep_prob=self.keep_prob)

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate 0.5 ~ 0.7 on training, but should be 1
            # for testing
            self.keep_prob = tf.placeholder(tf.float32)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])
            # img 28 x 28 x 1 (black/white)
            XImg = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            # L1 imgIn shape=(?, 28, 28, 1)
            # Conv          =(?, 28, 28, 32)
            # Pool          =(?, 14, 14, 32)
            L1 = self._getConvLayer(XImg, 3, 1, 32)

            # L2 imgIn shape=(?, 14, 14, 32)
            # Conv          =(?, 14, 14, 64)
            # Pool          =(?, 7, 7, 64)
            L2 = self._getConvLayer(L1, 3, 32, 64)

            # L3 imgIn shape=(?, 7, 7, 64)
            # Conv          =(?, 7, 7, 64)
            # Pool          =(?, 4, 4, 128)
            # reshape       =(?, 128 * 4 * 4)
            L3 = self._getConvLayer(L2, 3, 64, 128, True, 128 * 4 * 4)

            # L4 FC 128 * 4 * 4 inputs -> 625 outputs
            L4 = self._getFcLayer('W4', L3, 128 * 4 * 4, 625)

            # L5 Final FC 625 input -> outputs
            W5 = tf.get_variable('W5', shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random.normal([10]))
            self.logits = tf.matmul(L4, W5) + b5

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(self.cost)

        correctPredict = tf.equal( tf.argmax(self.logits, 1), tf.argmax(self.Y, 1) )
        self.accuracy = tf.reduce_mean(tf.cast(correctPredict, tf.float32))

    def predict(self, x, keep_prob=1.0):
        return self.sess.run(self.logits, feed_dict={self.X: x, self.keep_prob: keep_prob})

    def getAccuracy(self, x, y, keep_prob=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: x, self.Y: y, self.keep_prob: keep_prob})

    def train(self, x, y, keep_prob=0.7):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x, self.Y: y, self.keep_prob: keep_prob})

sess = tf.Session(config=config)
model1 = Model(sess, 'model1')

sess.run(tf.compat.v1.global_variables_initializer())

print('learning start')

for epoch in range(trainingEpochs):
    avgCost = 0

    for i in range(totalBatch):
        bx, by = mnist.train.next_batch(batchSize)
        c, _ = model1.train(bx, by)
        avgCost += c / totalBatch

    print('Epoch: {:04d}, cost: {:.9f}'.format((epoch+1), avgCost))

print('learning finish')

print('accuracy: {:.9f}'.format(model1.getAccuracy(mnist.test.images, mnist.test.labels)))
