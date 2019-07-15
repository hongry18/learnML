#!/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# multi variables linear
# H(x1, x2, x3) = w1x1 + w2x2 + w3x3 + b
# cost(W, b) = 1/m * sum(H(x1, x2, x3) - y)^2
# Matrix w1x1 + w2x2 + ... + wnxn
# Matrix multiplication

# H(X) = XW = (x1, x2, x3) * (w1; w2; w3) = (x1w1 + x2w2 + x3w3)
# Matrix 를 사용시x 인자를 앞에 사용해준다

def sample1():
    x1Data = [73., 93., 89., 96., 73.]
    x2Data = [80., 88., 91., 98., 66.]
    x3Data = [75., 93., 90., 100., 70.]
    yData = [152., 185., 180., 196., 142.]

    x1 = tf.placeholder(tf.float32)
    x2 = tf.placeholder(tf.float32)
    x3 = tf.placeholder(tf.float32)

    Y = tf.placeholder(tf.float32)

    w1 = tf.Variable(tf.random_normal([1]), name='weight1')
    w2 = tf.Variable(tf.random_normal([1]), name='weight2')
    w3 = tf.Variable(tf.random_normal([1]), name='weight3')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = (x1*w1) + (x2*w2) + (x3*w3) + b

    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        costVal, hyVal, _ = sess.run([cost, hypothesis, train], feed_dict={x1: x1Data, x2: x2Data, x3: x3Data, Y: yData})

        if step % 10 != 0:
            continue

        print(step, " - Cost: ", costVal, ", Prediction: ", hyVal)

# using Matrix
def sample2():
    dataX = [[73., 80., 75.], [93., 88., 93.], [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
    dataY = [[152.], [185.], [180.], [196.], [142.]]

    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([3,1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.matmul(X,W) + b


    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        costVal, hyVal, _ = sess.run([cost, hypothesis, train], feed_dict={X: dataX, Y: dataY})

        if step % 10 != 0:
            continue

        print(step, " - Cost: ", costVal, ",\nPrediction: ", hyVal)


# load from file
def sample3():
    xy = np.loadtxt('./lab-04-test-score.csv', delimiter=',', dtype=np.float32)
    dataX = xy[:, 0: -1]
    dataY = xy[:, [-1]]

    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([3,1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.matmul(X,W) + b

    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        costVal, hyVal, _ = sess.run([cost, hypothesis, train], feed_dict={X: dataX, Y: dataY})

        if step % 100 != 0:
            continue

        print(step, " - Cost: ", costVal, ",\nPrediction: ", hyVal)

def sample4():
    filenameQueue = tf.train.string_input_producer(['./lab-04-test-score.csv'], shuffle=False, name='filenameQueue')

    reader = tf.TextLineReader()
    key, val = reader.read(filenameQueue)

    recordDefaults = [[0.], [0.], [0.], [0.]]
    xy = tf.decode_csv(val, record_defaults=recordDefaults)

    trainXBatch, trainYBatch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([3,1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.matmul(X,W) + b

    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print(trainXBatch, trainYBatch)

    for step in range(2001):
        xBatch, yBatch = sess.run([trainXBatch, trainYBatch])
        """
        costVal, hyVal, _ = sess.run([cost, hypothesis, train], feed_dict={X: xBatch, Y: yBatch})

        if step % 100 != 0:
            continue

        print(step, " - Cost: ", costVal, ",\nPrediction: ", hyVal)
        """

#sample1()
#sample2()
#sample3()
sample4()
