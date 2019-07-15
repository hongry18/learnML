#!/bin/python
# -*- coding: utf-8 -*-

# hypothesis and cost
# linear regression
# simple hypothesis H(x) = Wx
import tensorflow as tf
import matplotlib.pyplot as plt

def sample1():
    x_data = [1,2,3]
    y_data = [1,2,3]

    W = tf.Variable(-3.0)
    #X = tf.placeholder(tf.float32)
    #Y = tf.placeholder(tf.float32)

    X = [1,2,3]
    Y = [1,2,3]

    hypothesis = X*W

    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    WVal = []
    costVal = []

    # minimize: gradient descent using derivative
    learningRate = 0.1
    gradient = tf.reduce_mean((W*X-Y)*X)
    descent = W - learningRate * gradient
    update = W.assign(descent)

    cost = tf.reduce_mean(tf.square(hypothesis -Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(cost)

    gvs = optimizer.compute_gradients(cost)
    applyGradients = optimizer.apply_gradients(gvs)
    
    sess = tf.Session()
    sess.run( tf.global_variables_initializer())

    """
    for i in range(-30, 50):
        feedW = i*0.1
        currCost, currW = sess.run([cost, W], feed_dict={W: feedW})
        WVal.append(currW)
        costVal.append(currCost)

    plt.plot(WVal, costVal)
    plt.show()
    """

    """
    for step in range(21):
        sess.run(update, feed_dict={X: x_data, Y: y_data})
        print(step, sess.run(cost, feed_dict={X: x_data, Y:y_data}), sess.run(W))
    """

    """
    for step in range(100):
        print(step, sess.run(W))
        sess.run(train)
    """

    for step in range(100):
        print(step, sess.run([gradient, W, gvs]))
        sess.run(applyGradients)


sample1()
