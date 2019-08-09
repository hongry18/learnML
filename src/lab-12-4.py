from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

tf.compat.v1.set_random_seed(777)

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

charSet = list(set(sentence))
charDic = {w: i for i, w in enumerate(charSet)}

dataDim = len(charSet)
hiddenSize = len(charSet)
numClasses = len(charSet)
sequenceLength = 10
learningRate = 1e-1

dataX = []
dataY = []
for i in range(0, len(sentence) - sequenceLength):
    xStr = sentence[i: i+sequenceLength]
    yStr = sentence[i+1: i+sequenceLength+1]

    x = [charDic[c] for c in xStr]
    y = [charDic[c] for c in yStr]

    dataX.append(x)
    dataY.append(y)

batchSize = len(dataX)

X = tf.placeholder(tf.int32, [None, sequenceLength])
Y = tf.placeholder(tf.int32, [None, sequenceLength])

xOneHot = tf.one_hot(X, numClasses)
# check out the shape
print(xOneHot)

multiCells = rnn.MultiRNNCell([rnn.BasicLSTMCell(hiddenSize, state_is_tuple=True) for _ in range(2)], state_is_tuple=True)

outputs, _states = tf.nn.dynamic_rnn(multiCells, xOneHot, dtype=tf.float32)

# FC Layer
X4fc = tf.reshape(outputs, [-1, hiddenSize])
outputs = tf.contrib.layers.fully_connected(X4fc, numClasses, activation_fn=None)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batchSize, sequenceLength, numClasses])

# weights
weights = tf.ones([batchSize, sequenceLength])
sequenceLoss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequenceLoss)
train = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss)

sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for i in range(500):
    _, l, results = sess.run([train, loss, outputs], feed_dict={X: dataX, Y: dataY})
    for j, result in enumerate(results):
        idx = np.argmax(result, axis=1)
        print(i, j, ''.join([charSet[t] for t in idx]), l)

results = sess.run(outputs, feed_dict={X: dataX})
for j, result in enumerate(results):
    idx = np.argmax(result, axis=1)
    if j is 0:
        print(''.join([charSet[t] for t in idx]), end='')
    else:
        print(charSet[idx[-1]], end='')
