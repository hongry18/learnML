import tensorflow as tf
import numpy as np

tf.compat.v1.set_random_seed(777)

sample = ' if you want you'
idx2char = list(set(sample))
char2idx = {c: i for i, c in enumerate(idx2char)}

dicSize = len(idx2char)
rnnHiddenSize = len(idx2char)
numClasses = len(idx2char)
batchSize = 1
sequenceLength = len(sample) - 1
learningRate = 1e-1

sampleIdx = [char2idx[c] for c in sample]
xData = [sampleIdx[:-1]]
yData = [sampleIdx[1:]]

X = tf.placeholder(tf.int32, [None, sequenceLength])
Y = tf.placeholder(tf.int32, [None, sequenceLength])

# flatten the data (ignore batches for now). no effect if the batch size is 1
xOneHot = tf.one_hot(X, numClasses)
x4softmax = tf.reshape(xOneHot, [-1, rnnHiddenSize])

# softmax layer ( rnn hidden size -> num classes )
softmaxW = tf.get_variable('softmax_W', [rnnHiddenSize, numClasses])
softmaxb = tf.get_variable('softmax_b', [numClasses])
outputs = tf.matmul(x4softmax, softmaxW) + softmaxb

# expend the data ( revive the batches )
outputs = tf.reshape(outputs, [batchSize, sequenceLength, numClasses])
weights = tf.ones([batchSize, sequenceLength])

# comput squence cost / loss
sequenceLoss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequenceLoss)
train = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for i in range(3000):
    l, _ = sess.run([loss, train], feed_dict={X: xData, Y: yData})
    result = sess.run(prediction, feed_dict={X: xData})

    resultStr = [idx2char[c] for c in np.squeeze(result)]
    print(i, ' loss: ', l, ', prediction: ', ''.join(resultStr))
