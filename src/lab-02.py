# -*- coding: utf-8 -*-
#!/bin/py
# linear Regression sample
# Hypothesis and cost function
# H(x) = Wx + b
# cost(W, b) = (for( int i=1; i<=m; i++ ) { sum(H(x[i]) - y[i])^2 }) / m

import tensorflow as tf

def sample1():
    # set x,y data
    xTrain = [1,2,3]
    yTrain = [1,2,3]

    W = tf.Variable(tf.random.normal([1]), name='weight')
    b = tf.Variable(tf.random.normal([1]), name='bias')

    # Out hypothesis = XW+b
    hypothesis = xTrain * W + b

    # cost/loss function
    cost = tf.reduce_mean( tf.square( hypothesis - yTrain) )

    # minimize
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost);

    # Launch the graph in a session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(train)
        if step % 20 == 0:
            print( step, sess.run(cost), sess.run(W), sess.run(b) )

def sample2():
    W = tf.Variable(tf.random.normal([1]), name='weight')
    b = tf.Variable(tf.random.normal([1]), name='bias')

    X = tf.placeholder(tf.float32, shape=[None])
    Y = tf.placeholder(tf.float32, shape=[None])

    hypothesis = X * W + b

    cost = tf.reduce_mean( tf.square( hypothesis- Y ))

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        vCost, vW, vb, _ = sess.run( [cost, W, b, train], feed_dict={X:[ 1, 2, 3, 4, 5], Y:[2.1, 3.1, 4.1, 5.1, 6.1]} )

        if step % 20 == 0:
            print( step, vCost, vW, vb )

    print( 'X=5 - ', sess.run( hypothesis, feed_dict={X:[5]} ) )
    print( 'X=2.5 - ', sess.run( hypothesis, feed_dict={X:[2.5]} ) )
    print( 'X=1.5, 3.5 - ', sess.run( hypothesis, feed_dict={X:[1.5, 3.5]} ) )

#sample1()
sample2()
