import tensorflow as tf
import numpy as np

tf.compat.v1.set_random_seed(777)

idx2char = ['h', 'i', 'e', 'l', 'o']
# Teach hello: hihell -> ihello
# hihell
xData = [[0, 1, 0, 2, 3, 3]]
xOneHot = [[
    [1, 0, 0, 0, 0]
    ,[0, 1, 0, 0, 0]
    ,[1, 0, 0, 0, 0]
    ,[0, 0, 1, 0, 0]
    ,[0, 0, 0, 1, 0]
    ,[0, 0, 0, 1, 0]
]]

# ihello
yData = [[1, 0, 2, 3, 3, 4]]

numClasses = 5
inputDim = 5
hiddenSize = 5
batchSize = 1
sequenceLength = 6
learningRate = 1e-1

X = tf.placeholder(tf.float32, [None, sequenceLength, inputDim])
Y = tf.placeholder(tf.int32, [None, sequenceLength])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hiddenSize, state_is_tuple=True)
initialState = cell.zero_state(batchSize, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initialState, dtype=tf.float32)

# FC Layer
X4fc = tf.reshape(outputs, [-1, hiddenSize])
# fcW = tf.get_variable('fc_w', [hiddenSize, numClasses])
# fcb = tf.get_variable('fc_b', [numClasses])
# outputs = tf.matmul(X4fc, fcW) + fcb
outputs = tf.contrib.layers.fully_connected(inputs=X4fc, num_outputs=numClasses, activation_fn=None)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batchSize, sequenceLength, numClasses])

weights = tf.ones([batchSize, sequenceLength])
sequenceLoss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequenceLoss)
train = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)
sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for i in range(50):
    l, _ = sess.run([loss, train], feed_dict={X: xOneHot, Y: yData})
    result = sess.run(prediction, feed_dict={X: xOneHot})
    print(i, ' loss: ', l, ', prediction: ', result, ' true Y: ', yData)

    # print char using dic
    resultStr = [idx2char[c] for c in np.squeeze(result)]
    print('prediction str: ', ''.join(resultStr))
