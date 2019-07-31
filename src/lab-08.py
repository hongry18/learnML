import tensorflow as tf
import numpy as np
import pprint
tf.compat.v1.set_random_seed(777)

pp = pprint.PrettyPrinter(indent=4)
sess = tf.compat.v1.InteractiveSession()

print('# array slicing')
t = np.array([0., 1., 2., 3., 4., 5., 6.])
pp.pprint(t)
print(t.ndim)
print(t.shape)
print(t[0], t[1], t[2])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])

t = np.array([[1.,2.,3.], [4.,5.,6.], [7., 8., 9.], [10., 11., 12.]])
pp.pprint(t)
print(t.ndim)
print(t.shape)

t = tf.constant([1,2,3,4])
r = tf.shape(t)
pp.pprint(r)
pp.pprint(r.eval())

t = tf.constant([[1,2],[3,4]])
r = tf.shape(t)
pp.pprint(r)
pp.pprint(r.eval())

t = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
r = tf.shape(t)
pp.pprint(r)
pp.pprint(r.eval())

# matmul
print('# matmul')
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
r = tf.matmul(matrix1, matrix2)
pp.pprint(r.eval())

# broadcasting
# multiply
print('# broadcasting multiply, sum')
r = matrix1 * matrix2
pp.pprint(r.eval())

r = matrix1 + matrix2
pp.pprint(r.eval())

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])
r = matrix1 + matrix2
pp.pprint(r.eval())

# random values for variable initializations
print('# random values for variable initializations')
r = tf.random.normal([3])
pp.pprint(r.eval())

r = tf.random.normal([2])
pp.pprint(r.eval())

r = tf.random.normal([2,3])
pp.pprint(r.eval())

# reduce mean / sum
print('# reduce mean/sum')
r = tf.reduce_mean([1,2], axis=0)
print(r.eval())

x = [[1., 2.], [3., 4.]]
r = tf.reduce_mean(x)
print(r.eval())

r = tf.reduce_mean(x, axis=0)
print(r.eval())

r = tf.reduce_mean(x, axis=1)
print(r.eval())

r = tf.reduce_mean(x, axis=-1)
print(r.eval())

r = tf.reduce_sum(x)
print(r.eval())

r = tf.reduce_sum(x, axis=0)
print(r.eval())

r = tf.reduce_sum(x, axis=1)
print(r.eval())

r = tf.reduce_sum(x, axis=-1)
print(r.eval())

# argmax with axis
print('# argmax with axis')
x = [[0,1,2], [2,1,0]]
r = tf.argmax(x).eval()
print(r)

r = tf.argmax(x, axis=0).eval()
print(r)

r = tf.argmax(x, axis=1).eval()
print(r)

r = tf.argmax(x, axis=-1).eval()
print(r)

# reshape, squeeze, expand_dims
print('# reshape, squeeze, expand_dims')
t = np.array( [[[0,1,2],[3,4,5]], [[6,7,8],[9,10,11]]] )
print(t.shape)

## origin print
pp.pprint(t)

r = tf.reshape(t, shape=[-1, 3]).eval()
print(r)

r = tf.reshape(t, shape=[-1, 1, 3]).eval()
print(r)

r = tf.squeeze( [[0], [1], [2]] ).eval()
print(r)

r = tf.expand_dims([0,1,2], 1).eval()
print(r)

# one hot
print('# one hot')

r = tf.one_hot([[0], [1], [2], [0]], depth=3).eval()
print(r)

t = tf.one_hot([[0], [1], [2], [0]], depth=3)
r = tf.reshape(t, shape=[-1, 3]).eval()
print(r)

# casting
print('# casting')
r = tf.cast([1.9, 2.4, 3.1, 4.4449, 5.0001, 6.50000001], tf.int32).eval()
print(r)

r = tf.cast([True, False, 1==1, 1!=1, 0==1], tf.int32).eval()
print(r)

# stack
print('# stack')
x = [1,4]
y = [2,5]
z = [3,6]

r = tf.stack([x,y,z]).eval()
print(r)

r = tf.stack([x,y,z], axis=1).eval()
print(r)

# ones like and zeros like
print('# ones like, zeros like')
x = [[0,1,2], [2,1,0]]
r = tf.ones_like(x).eval()
print(r)

r = tf.zeros_like(x).eval()
print(r)

# zip
print('# zip')
for x, y in zip([1,2,3], [4,5,6]):
    print(x, y)

for x, y, z in zip([1,2,3], [4,5,6], [7,8,8]):
    print(x, y, z)

# transpose
print('# transpose')
t = np.array([[[0,1,2], [3,4,5]], [[6,7,8], [9,10,11]]])
pp.pprint(t.shape)
pp.pprint(t)

t1 = tf.transpose(t, [1, 0, 2])
pp.pprint(sess.run(t1).shape)
pp.pprint(sess.run(t1))

t = tf.transpose(t1, [1,0,2])
pp.pprint(sess.run(t).shape)
pp.pprint(sess.run(t))

t2 = tf.transpose(t, [1, 2, 0])
pp.pprint(sess.run(t2).shape)
pp.pprint(sess.run(t2))

t = tf.transpose(t2, [2, 0, 1])
pp.pprint(sess.run(t).shape)
pp.pprint(sess.run(t))
