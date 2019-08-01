import tensorflow as tf

tf.compat.v1.set_random_seed(777)

xData = [[1.], [2.], [3.]]
yData = [[1.], [2.], [3.]]

X = tf.placeholder(tf.float32, [None, 1])
Y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.truncated_normal([1,1]))
b = tf.Variable(5.)

hypothesis = tf.matmul(X, W) + b

assert hypothesis.shape.as_list() == Y.shape.as_list()
diff = (hypothesis - Y)

d_l1 = diff
d_b = d_l1
d_w = tf.matmul(tf.transpose(X), d_l1)

print(X, W, d_l1, d_w)

learning_rate = 0.1
step = [ tf.assign(W, W-learning_rate * d_w), tf.assign(b, b-learning_rate * tf.reduce_mean(d_b)) ]

RMSE = tf.reduce_mean(tf.square(Y-hypothesis))

sess = tf.InteractiveSession()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

for i in range(1001):
    if i % 100 != 0:
        continue
    print(i, sess.run([step, RMSE], feed_dict={X:xData, Y:yData}))

print(sess.run(hypothesis, feed_dict={X:xData}))
