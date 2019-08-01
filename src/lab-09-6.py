import tensorflow as tf

tf.compat.v1.set_random_seed(777)

xData = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
yData = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

X = tf.placeholder(tf.float32, [None, 3])
Y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.truncated_normal([3,1]))
b = tf.Variable(5.)

hypothesis = tf.matmul(X, W) + b
print(hypothesis.shape, Y.shape)

# diff
assert hypothesis.shape.as_list() == Y.shape.as_list()
diff = (hypothesis - Y)

d_l1 = diff
d_b = d_l1
d_w = tf.matmul(tf.transpose(X), d_l1)

print(X, W, d_l1, d_w)

learning_rate = 1e-6
step = [ tf.assign(W, W-learning_rate * d_w), tf.assign(b, b-learning_rate * tf.reduce_mean(d_b)) ]

RMSE = tf.reduce_mean(tf.square(Y-hypothesis))

sess = tf.InteractiveSession()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

for i in range(10001):
    if i % 100 != 0:
        continue
    print(i, sess.run([step, RMSE], feed_dict={X:xData, Y:yData}))

print(sess.run(hypothesis, feed_dict={X:xData}))
