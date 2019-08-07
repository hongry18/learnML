# rnn
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

sess = tf.InteractiveSession()

# one hot encoding
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

def in1():
    # one cell RNN input_dim(4) -> output_dim (2)
    with tf.variable_scope('one_cell') as scope:
        hiddenSize = 2
        cell = tf.keras.layers.SimpleRNNCell(units=hiddenSize)
        print(cell.output_size, cell.state_size)

        xData = np.array([[h]], dtype=np.float32)
        print(xData)
        outputs, _states = tf.nn.dynamic_rnn(cell, xData, dtype=tf.float32)

        sess.run(tf.global_variables_initializer())
        print(outputs.eval())

def in2():
    # one cell RNN input_dim(4) -> output_dim(2). sequence: 5
    with tf.variable_scope('two_sequence') as scope:
        hiddenSize = 2
        cell = tf.keras.layers.SimpleRNNCell(units=hiddenSize)
        xData = np.array([[h, e, l, l, o]], dtype=np.float32)
        print(xData.shape)
        print(xData)
        outputs, _states = tf.nn.dynamic_rnn(cell, xData, dtype=tf.float32)
        sess.run(tf.global_variables_initializer())
        print(outputs.eval())

def in3():
    # one cell RNN input_dim(4) -> output_dim(2). sequence: 5, batch: 3
    # 3 batches 'hello', 'eolll', 'lleel'
    with tf.variable_scope('3_batches') as scope:
        xData = np.array([[h, e, l, l, o], [e, o, l, l, l], [l, l, e, e, l]], dtype=np.float32)
        print(xData)

        hiddenSize = 2
        cell = tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True)
        outputs, _states = tf.nn.dynamic_rnn(cell, xData, dtype=tf.float32)
        sess.run(tf.compat.v1.global_variables_initializer())
        print(outputs.eval())

def in4():
    # one cell RNN input_dim(4) -> output_dim(5). sequence: 5, batch: 3
    # 3 batches 'hello', 'eolll', 'lleel'
    with tf.variable_scope('3_batches_dynamic_length') as scope:
        xData = np.array([[h, e, l, l, o], [e, o, l, l, l], [l, l, e, e, l]], dtype=np.float32)
        print(xData)

        hiddenSize = 2
        cell = tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True)
        outputs, _states = tf.nn.dynamic_rnn(cell, xData, sequence_length=[5,3,4], dtype=tf.float32)
        sess.run(tf.compat.v1.global_variables_initializer())
        print(outputs.eval())

def in5():
    with tf.variable_scope('initial_state') as scope:
        batchSize = 3
        xData = np.array([[h, e, l, l, o], [e, o, l, l, l], [l, l, e, e, l]], dtype=np.float32)
        print(xData)

        # one cell RNN input_dim(4) -> output_dim(5). sequence: 5, batch: 3
        hiddenSize = 2
        cell = tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True)
        initial_state = cell.zero_state(batchSize, tf.float32)
        outputs, _states = tf.nn.dynamic_rnn(cell, xData, initial_state=initial_state, dtype=tf.float32)
        sess.run(tf.compat.v1.global_variables_initializer())
        print(outputs.eval())

def in6():
    # create input data
    batchSize = 3
    sequenceLength = 5
    inputDim = 3
    xData = np.arange(45, dtype=np.float32).reshape(batchSize, sequenceLength, inputDim)
    print(xData)

    # one cell RNN input_dim(3) -> output_dim(5). sequence: 5, batch: 3
    with tf.variable_scope('generated_data') as scope:
        cell = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)
        initial_state = cell.zero_state(batchSize, tf.float32)
        outputs, _states = tf.nn.dynamic_rnn(cell, xData, initial_state=initial_state, dtype=tf.float32)
        sess.run(tf.compat.v1.global_variables_initializer())
        print(outputs.eval())

def in7():
    # create input data
    batchSize = 3
    sequenceLength = 5
    inputDim = 5
    dataSize = batchSize * sequenceLength * inputDim
    xData = np.arange(dataSize, dtype=np.float32).reshape(batchSize, sequenceLength, inputDim)
    print(xData)

    # Make rnn
    with tf.variable_scope('MultiRNNCell') as scope:
        cell = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)
        cell = rnn.MultiRNNCell([cell] * 3, state_is_tuple=True) # 3 Layers

        # rnn in/out
        outputs, _states = tf.nn.dynamic_rnn(cell, xData, dtype=tf.float32)
        print('Dynamic rnn: ', outputs)
        sess.run(tf.compat.v1.global_variables_initializer())
        print(outputs.eval())

in7()
