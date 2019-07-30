#!/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# Classification
# 0, 1 encoding (ham, spam), (show, hide)...

# g(z) -> 0 < val < 1 = 1/(1+e^-z) = (sigmoid, logistic) function
# z = WX
# H(x) = g(z)

def sample1():
    xData = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]
    yData = [[0], [0], [0], [1], [1], [1]]

    X = tf.placeholder(tf.float32, shape=[None,2])
    Y = tf.placeholder(tf.float32, shape=[None,1])
    W = tf.Variable(tf.random_normal([2,1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.sigmoid(tf.matmul(X,W) +b)

    cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

    train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(10001):
            costVal, _ = sess.run([cost, train], feed_dict={X:xData, Y:yData})
            if step % 200 != 0:
                continue

            print(step, costVal)

        h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:xData, Y:yData})
        print("hypothesys: ", h, "\npredicted: ", c, "\naccuracy: ", a)

def sample2():
    xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
    xData = xy[:, 0:-1]
    yData = xy[:, [-1]]

    X = tf.placeholder(tf.float32, shape=[None,8])
    Y = tf.placeholder(tf.float32, shape=[None,1])
    W = tf.Variable(tf.random_normal([8, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.sigmoid(tf.matmul(X,W) +b)

    cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

    train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(10001):
            costVal, _ = sess.run([cost, train], feed_dict={X:xData, Y:yData})
            if step % 200 != 0:
                continue

            print(step, costVal)

        h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:xData, Y:yData})
        print("hypothesys: ", h, "\npredicted: ", c, "\naccuracy: ", a)



if __name__ == '__main__':
    #sample1()
    sample2()
