import tensorflow as tf
import numpy as np

tf.compat.v1.set_random_seed(777)

sample = ' if you want you'
idx2char = list(set(sample)) # index - > char
char2idx = {c: i for i, c in enumerate(idx2char)} # char -> index

# hyper parameters
# RNN input size (one hot size)
dicSize = len(char2idx)
# RNN output size
hiddenSize = len(char2idx)
# final output size (RNN or softmax, etc.)
numClasses = len(char2idx)
# one sample data, one batch
batchSize = 1
# number of lstm rollings (unit #)
sequenceLength = len(sample) - 1
learningRate = 1e-1

sampleIdx = [char2idx[c] for c in sample]
# X data sample (0 ~ n-1) hello: hell
xData = [sampleIdx[:-1]]
# Y data sample (1 ~ n) hello: ello
yData = [sampleIdx[1:]]

X = tf.placeholder(tf.int32, [None, sequenceLength])
Y = tf.placeholder(tf.int32, [None, sequenceLength])

# one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
xOneHot = tf.one_hot(X, numClasses)
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hiddenSize, state_is_tuple=True)
initialState = cell.zero_state(batchSize, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, xOneHot, initial_state=initialState, dtype=tf.float32)

# FC Layer
X4fc = tf.reshape(outputs, [-1, hiddenSize])
outputs = tf.contrib.layers.fully_connected(X4fc, numClasses, activation_fn=None)

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
    l, _ = sess.run([loss, train], feed_dict={X: xData, Y: yData})
    r = sess.run(prediction, feed_dict={X: xData})

    # print char uisng dic
    rStr = [idx2char[c] for c in np.squeeze(r)]
    print(i, ' loss: ', l, ', prediction: ', ''.join(rStr))
