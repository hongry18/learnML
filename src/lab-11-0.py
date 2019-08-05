# basic cnn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.compat.v1.InteractiveSession()
img = np.array( [[ [[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]] ]], dtype=np.float32)
print(img.shape)

def sample1():
    plt.imshow(img.reshape(3,3), cmap='gray')

def sample2():
    weight = tf.constant( [[ [[1.]], [[1.]]], [[[1.]], [[1.]] ]])

    print(weight.shape)

    c2d = tf.nn.conv2d(img, weight, strides=[1,1,1,1], padding='VALID')
    c2di = c2d.eval()
    print(c2di.shape)
    c2di = np.swapaxes(c2di, 0, 3)

    for i, oneImg in enumerate(c2di):
        print( oneImg.reshape(2, 2) )
        plt.subplot(1, 2, i+1)
        plt.imshow(oneImg.reshape(2,2), cmap='gray')

def sample3():
    weight = tf.constant( [[ [[1.]], [[1.]]], [[[1.]], [[1.]] ]])
    print(weight.shape)

    c2d = tf.nn.conv2d(img, weight, strides=[1,1,1,1], padding='SAME')
    c2di = c2d.eval()
    print(c2di.shape)
    c2di = np.swapaxes(c2di, 0, 3)

    for i, oImg in enumerate(c2di):
        print(oImg.reshape(3,3))
        plt.subplot(1, 2, i+1)
        plt.imshow(oImg.reshape(3,3), cmap='gray')

def sample4():
    # filters (2,2,1,3)
    weight = tf.constant(
        [
            [
                [[1., 10., -1.]]
                ,[[1., 10., -1.]]
            ]
            ,[
                [[1., 10., -1.]]
                ,[[1., 10., -1.]]
            ]
        ]
    )
    print(weight.shape)

    c2d = tf.nn.conv2d(img, weight, strides=[1,1,1,1], padding='SAME')
    c2di = c2d.eval()
    print(c2di.shape)
    c2di = np.swapaxes(c2di, 0, 3)

    for i, oi in enumerate(c2di):
        print(oi.reshape(3, 3))
        plt.subplot(1, 3, i+1)
        plt.imshow(oi.reshape(3,3), cmap='gray')

def sample5():
    # max pooling
    img = np.array(
        [[
            [[4], [3]]
            ,[[2], [1]]
        ]]
        ,dtype=np.float32
    )
    pool = tf.nn.max_pool(img, ksize=[1,2,2,1], strides=[1,1,1,1], padding='VALID')
    print(pool.shape)
    print(pool.eval())

def sample6():
    img = np.array(
        [[
            [[4], [3]]
            ,[[2], [1]]
        ]]
        ,dtype=np.float32
    )

    pool = tf.nn.max_pool(img, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
    print(pool.shape)
    print(pool.eval())

def sample7():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

    img = mnist.train.images[0].reshape(28, 28)
    plt.imshow(img, cmap='gray')

    img = img.reshape(-1, 28, 28, 1)
    W1 = tf.Variable(tf.random.normal([3, 3, 1, 5], stddev=0.01))
    c2d = tf.nn.conv2d(img, W1, strides=[1,2,2,1], padding='SAME')
    print(c2d)

    pool = tf.nn.max_pool(c2d, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    sess.run(tf.compat.v1.global_variables_initializer())

    c2di = c2d.eval()
    c2di = np.swapaxes(c2di, 0, 3)
    for i, oi in enumerate(c2di):
        plt.subplot(2, 5, i+1)
        plt.imshow(oi.reshape(14, 14), cmap='gray')
        pass

    pImg = pool.eval()
    pImg = np.swapaxes(pImg, 0, 3)
    for i, oi in enumerate(pImg):
        plt.subplot(2, 5, i+6)
        plt.imshow(oi.reshape(7, 7), cmap='gray')
        pass


sample7()
plt.show()
