# Lab 9 XOR
import tensorflow as tf
import numpy as np

tf.compat.v1.set_random_seed(777)

xData = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
yData = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.random.normal([2,1], name='weight'))
b = tf.Variable(tf.random.normal([1], name='bais'))

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean( Y*tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    _, cv, wv = sess.run([train, cost, W], feed_dict={X:xData, Y:yData})
    if step % 1000 != 0:
        continue
    print(step, cv, wv)

h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: xData, Y: yData})
print("hypothesis:\n", h, "\npredicted:\n", c, "\naccuracy:\n", a)
