#!/bin/python3
# -*- coding: utf-8 -*-
# Lab 7 Learning rate and Evaluation

import tensorflow as tf
tf.set_random_seed(777)

xData = [
    [1,2,1]
    ,[1,3,2]
    ,[1,3,4]
    ,[1,5,5]
    ,[1,7,5]
    ,[1,2,5]
    ,[1,6,6]
    ,[1,7,7]
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

xTest = [
    [2,1,1]
    ,[3,1,2]
    ,[3,3,4]
]

yTest = [
    [0,0,1]
    ,[0,0,1]
    ,[0,0,1]
]

learningRate = 1e-1

X = tf.placeholder(tf.float32, [None, 3])
Y = tf.placeholder(tf.float32, [None, 3])

W = tf.Variable(tf.random_normal([3, 3]))
b = tf.Variable(tf.random_normal([3]))

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost = tf.reduce_mean( -tf.reduce_sum( Y * tf.log(hypothesis), axis=1 ) )
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
isCorrect = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(201):
    costVal, WVal, _ = sess.run([cost, W, optimizer], feed_dict={X: xData, Y: yData})
    print(step, costVal, WVal)

print("Prediction: ", sess.run(prediction, feed_dict={X: xTest}))
print("Accuracy: ", sess.run(accuracy, feed_dict={X: xTest, Y: yTest}))
