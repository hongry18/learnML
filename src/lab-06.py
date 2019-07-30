#!/bin/python3
# -*- coding: utf-8 -*-
# softmax classification

import tensorflow as tf
import numpy as np

## hypothesis = tf.nn.softmax( tf.matmul(X, W) + b )

# Cross entropy cost/loss
## cost = tf.reduce_mean( -tf.reduce_sum( Y * tf.log(hypothesis), axis=1 ))
## optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

def sample1():
    xData = [
        [1,2,1,1]
        ,[2,1,3,2]
        ,[3,1,3,4]
        ,[4,1,5,5]
        ,[1,7,5,5]
        ,[1,2,5,6]
        ,[1,6,6,6]
        ,[1,7,7,7]
    ]

    yData = [
        [0,0,1]
        ,[0,0,1]
        ,[0,0,1]
        ,[0,1,0]
        ,[0,1,0]
        ,[0,1,0]
        ,[1,0,0]
        ,[1,0,0]
    ]

    X = tf.placeholder('float', [None, 4])
    Y = tf.placeholder('float', [None, 3])
    nbClasses = 3

    W = tf.Variable(tf.random_normal([4, nbClasses]), name='weight')
    b = tf.Variable(tf.random_normal([nbClasses]), name='bias')

    # tf.nn.softmax computes softmax activations
    # softmax = exp(logits) / reduce_sum(exp(logits), dim)
    hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

    # cross entropy cost / loss
    cost = tf.reduce_mean( -tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

    # launch graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(2001):
            sess.run(optimizer, feed_dict={X: xData, Y: yData})
            if step % 200 != 0:
                continue

            print(step, sess.run(cost, feed_dict={X: xData, Y: yData}))

        a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
        print(a, sess.run(tf.argmax(a, 1)))

        multiple = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
        print(multiple, sess.run(tf.argmax(multiple, 1)))

# fancy softmax classifier
# cross entropy, ont_hot, reshape
def sample2():
    tf.set_random_seed(777)
    xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
    xData = xy[:, 0: -1]
    yData = xy[:, [-1]]

    print( xData.shape, yData.shape )
    '''
    (101, 16) (101, 1)
    '''

    nbClasses = 7

    X = tf.placeholder(tf.float32, [None, 16])
    Y = tf.placeholder(tf.int32, [None, 1])

    yOneHot = tf.one_hot(Y, nbClasses)
    print('one hot: ', yOneHot)
    yOneHot = tf.reshape(yOneHot, [-1, nbClasses])
    print('reshape one hot: ', yOneHot)

    '''
        one_hot: Tensor("one_hot: 0", shape=(?, 1, 7), dtype=float32)
        reshape one_hot: Tensor("Reshape: 0", shape=(?, 7), dtype=float32)
    '''

    W = tf.Variable(tf.random_normal([16, nbClasses], name='weight'))
    b = tf.Variable(tf.random_normal([nbClasses], name='bias'))

    logits = tf.matmul(X,W) + b
    hypothesis = tf.nn.softmax(logits)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.stop_gradient([yOneHot])))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

    prediction = tf.argmax(hypothesis, 1)
    correctPrediction = tf.equal(prediction, tf.argmax(yOneHot, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

    # launch graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(2001):
            _, costVal, accVal = sess.run([optimizer, cost, accuracy], feed_dict={X: xData, Y: yData})

            if step % 100 != 0:
                continue

            print("step: ", step, ", cost: ", costVal, ", accuracy: ", accVal)

        pred = sess.run(prediction, feed_dict={X:xData})
        for p, y in zip(pred, yData.flatten()):
            print("[{}] Prediction: {} True Y: {}".format(p==int(y), p, int(y)))

if __name__ == '__main__':
    #sample1()
    sample2()
