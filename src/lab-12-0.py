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
        cell = rnn.MultiRNNCell([cell], state_is_tuple=True) # 3 Layers
        #cells = [tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True) for _ in range(3)]
        #cell = rnn.MultiRNNCell(cells)

        # rnn in/out
        outputs, _states = tf.nn.dynamic_rnn(cell, xData, dtype=tf.float32)
        print('Dynamic rnn: ', outputs)
        sess.run(tf.compat.v1.global_variables_initializer())
        print(outputs.eval())

def in8():
    batchSize = 3
    sequenceLength = 5
    inputDim = 3
    dataSize = batchSize * sequenceLength * inputDim
    xData = np.arange(dataSize, dtype=np.float32).reshape(batchSize, sequenceLength, inputDim)
    with tf.variable_scope('dynamic_rnn') as scope:
        cell = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)
        outputs, _states = tf.nn.dynamic_rnn(cell, xData, dtype=tf.float32, sequence_length=[1, 3, 2])

        # length 1 for batch 1, length 2 for batch 2

        print('dynamic rnn: ', outputs)
        sess.run(tf.compat.v1.global_variables_initializer())
        print(outputs.eval())

def in9():
    batchSize = 3
    sequenceLength = 5
    inputDim = 3
    dataSize = batchSize * sequenceLength * inputDim
    xData = np.arange(dataSize, dtype=np.float32).reshape(batchSize, sequenceLength, inputDim)
    with tf.variable_scope('bi-directional') as scope:
        cellFw = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)
        cellBw = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cellFw, cellBw, xData, sequence_length=[2, 3, 1], dtype=tf.float32)
        sess.run(tf.compat.v1.global_variables_initializer())
        print(sess.run(outputs))
        print(sess.run(states))

def in10():
    hiddenSize = 3
    sequenceLength = 5
    inputDim = 3
    batchSize = 3
    numClasses = 5
    dataSize = batchSize * sequenceLength * inputDim

    xData = np.arange(dataSize, dtype=np.float32).reshape(batchSize, sequenceLength, inputDim)
    print(xData)
    xData = xData.reshape(-1, hiddenSize)
    print(xData)

    softmaxW = np.arange(15, dtype=np.float32).reshape(hiddenSize, numClasses)
    outputs = np.matmul(xData, softmaxW)
    outputs = outputs.reshape(-1, sequenceLength, numClasses)
    print(outputs)

def in11():
    # [batchSize, sequenceLength]
    yData = tf.constant([[1, 1, 1]])

    # [batchSize, sequenceLength, embDim]
    prediction1 = tf.constant([[[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]]], dtype=tf.float32)
    prediction2 = tf.constant([[[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]], dtype=tf.float32)

    prediction3 = tf.constant([[[1, 0], [1, 0], [1, 0]]], dtype=tf.float32)
    prediction4 = tf.constant([[[0, 1], [0, 1], [0, 1]]], dtype=tf.float32)

    # [batchSize * sequenceLength]
    weights =tf.constant([[1, 1, 1]], dtype=tf.float32)

    sequenceLoss1 = tf.contrib.seq2seq.sequence_loss(prediction1, yData, weights)
    sequenceLoss2 = tf.contrib.seq2seq.sequence_loss(prediction2, yData, weights)
    sequenceLoss3 = tf.contrib.seq2seq.sequence_loss(prediction3, yData, weights)
    sequenceLoss4 = tf.contrib.seq2seq.sequence_loss(prediction4, yData, weights)

    sess.run(tf.compat.v1.global_variables_initializer())
    print('loss1: ', sequenceLoss1.eval())
    print('loss2: ', sequenceLoss2.eval())
    print('loss3: ', sequenceLoss3.eval())
    print('loss4: ', sequenceLoss4.eval())

in7()
