import tensorflow as tf
import numpy as np
import matplotlib
import os

tf.compat.v1.set_random_seed(777)

#if 'DISPLAY' not in os.environ:
#    matplotlib.use('Agg')

import matplotlib.pyplot as plt

def minMaxScaler(d):
    numerator = d - np.min(d, 0)
    denominator = np.max(d, 0) - np.min(d, 0)
    return numerator / (denominator + 1e-7)

seqLen = 7
dataDim = 5
hiddenDim = 10
outputDim = 1
learningRate = 1e-2
iterations = 500

xy = np.loadtxt('./data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1]

trainSize = int(len(xy) * 0.7)
trainSet = xy[0:trainSize]
testSet = xy[trainSize - seqLen:]

trainSet = minMaxScaler(trainSet)
testSet = minMaxScaler(testSet)

def buildDataset(timeSeries, seqLen):
    dX = []
    dY = []
    for i in range(0, len(timeSeries) - seqLen):
        _x = timeSeries[i: i+seqLen, :]
        _y = timeSeries[i+seqLen, [-1]]
        dX.append(_x)
        dY.append(_y)

    return np.array(dX), np.array(dY)

trainX, trainY = buildDataset(trainSet, seqLen)
testX, testY = buildDataset(testSet, seqLen)

# input placeholders
X = tf.placeholder(tf.float32, [None, seqLen, dataDim])
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
cells = [tf.contrib.rnn.BasicLSTMCell(num_units=hiddenDim, state_is_tuple=True, activation=tf.tanh) for _ in range(2)]
cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
yPred = tf.contrib.layers.fully_connected(outputs[:, -1], outputDim, activation_fn=None)

# cost/loss
loss = tf.reduce_mean(tf.square(yPred - Y))
# optimizer
optimizer = tf.train.AdamOptimizer(learningRate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# training step
for i in range(iterations):
    _, stepLoss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
    print('[step: {}] loss: {}'.format(i, stepLoss))

# test step
testPredict = sess.run(yPred, feed_dict={X: testX})
rmseVal = sess.run(rmse, feed_dict={targets: testY, predictions: testPredict})
print('RMSE: {}'.format(rmseVal))

plt.plot(testY)
plt.plot(testPredict)
plt.xlabel("time period")
plt.ylabel("stock price")
plt.show()
