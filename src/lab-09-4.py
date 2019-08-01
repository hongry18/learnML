# Lab  9 XOR tensorboard
import tensorflow as tf
import numpy as np

tf.compat.v1.set_random_seed(777)

xData = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
yData = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

with tf.name_scope('Layer1'):
    W1 = tf.Variable(tf.random.normal([2, 2]), name='weight1')
    b1 = tf.Variable(tf.random.normal([2]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    tf.summary.histogram("W1", W1)
    tf.summary.histogram("b1", b1)
    tf.summary.histogram("Lyaer1", layer1)

with tf.name_scope('Layer2'):
    W2 = tf.Variable(tf.random.normal([2, 2]), name='weight2')
    b2 = tf.Variable(tf.random.normal([2]), name='bias2')
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    tf.summary.histogram("W2", W2)
    tf.summary.histogram("b2", b2)
    tf.summary.histogram("Lyaer2", hypothesis)

with tf.name_scope('Cost'):
    cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
    tf.summary.histogram('Cost', cost)

with tf.name_scope('Train'):
    train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
tf.summary.scalar("accuracy", accuracy)

sess = tf.Session()
mergedSummary = tf. summary.merge_all()
writer = tf.summary.FileWriter("./logs/xor_logs_r0_01")
writer.add_graph(sess.graph)

sess.run(tf.global_variables_initializer())

for step in range(10001):
    _, summary, cv = sess.run([train, mergedSummary, cost], feed_dict={X:xData, Y:yData})
    writer.add_summary(summary, global_step=step)
    if step % 1000 != 0:
        continue

    print(step, cv)

h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:xData, Y:yData})
print(f"\nHypothesis:\n{h}\nPredicted:\n{p}\nAccuracy:\n{a}")
