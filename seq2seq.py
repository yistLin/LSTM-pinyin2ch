#!/usr/bin/env python3
from __future__ import print_function
import os, sys
import numpy as np
import random
import string
import tensorflow as tf

# get train data and valid data filename
try:
    vocab_filename = sys.argv[1]
    train_data_filename = sys.argv[2]
    valid_data_filename = sys.argv[3]
except Exception:
    print('Cannot get data')
    sys.exit(1)

# open training file
train_data_f = open(train_data_filename, 'r')
valid_data_f = open(valid_data_filename, 'r')
vocab_f = open(vocab_filename, 'r')

# all pinyin and 0-9
char2id = {}
id2char = []
cnt = 0
for line in vocab_f:
    v = line.strip()
    id2char.append(v)
    char2id[v] = cnt
    cnt += 1

vocab_size = len(id2char)
ch_vocab_size = 20000
padding_length = 15

# train data and valid data
# line_sp[0] is chinese characters as label
# line_sp[1] is pinyin characters as input
ch_dict = {}
ch_index = 0
train_list = []
ch_dict['_PAD'] = 0

# build chinese characters mapping
for line in train_data_f:
    line_sp = line.strip('\n').split('\t')
    raw_pinyin_list = line_sp[1].split()
    if 3 <= len(raw_pinyin_list) <= 15:
        raw_ch_list = line_sp[0].split()
        for ch in raw_ch_list:
            if ch not in ch_dict:
                ch_index += 1
                ch_dict[ch] = ch_index
ch_vocab_size = len(ch_dict)

train_data_f.seek(0)
for line in train_data_f:
    line_sp = line.strip('\n').split('\t')
    raw_pinyin_list = line_sp[1].split()
    raw_ch_list = line_sp[0].split()
    if 3 <= len(raw_pinyin_list) <= 15:
        raw_pinyin_list += ['_PAD'] * (padding_length - len(raw_pinyin_list))
        pinyin_list = np.zeros(shape=(len(raw_pinyin_list), vocab_size), dtype=np.float)
        raw_ch_list += ['_PAD'] * (padding_length - len(raw_ch_list))
        ch_list = np.zeros(shape=(len(raw_ch_list), ch_vocab_size), dtype=np.float)
        for (i, word) in enumerate(raw_pinyin_list):
            pinyin_list[i, char2id[word]] = 1.0
        for (i, word) in enumerate(raw_ch_list):
            ch_list[i, ch_dict[word]] = 1.0
        train_list.append( (pinyin_list, ch_list) )

valid_list = []
for line in valid_data_f:
    line_sp = line.strip('\n').split('\t')
    pinyin_list = list(map(lambda x: char2id[x], line_sp[1].split()))
    ch_list = line_sp[0].split()
    valid_list.append( (pinyin_list, ch_list, len(pinyin_list)) )

# close training file
vocab_f.close()
train_data_f.close()
valid_data_f.close()

# testing vocab
print('testing vocab:')
print('char2id{}:', char2id['song'], char2id['zi'], char2id['wei'], char2id['9'])
print('id2char[]:', id2char[1], id2char[26], id2char[10])

batch_size=64
num_unrollings=10

class BatchGenerator(object):
    def __init__(self, data, batch_size):
        self._data = data
        self._batch_size = batch_size
        self.start = 0
        self.end = 0
        self._length = len(data)
        self.batch = []
        self.prepare_batch()

    def prepare_batch(self):
        while True:
            # print(self.start, self.end, self._length)
            if self.end + self._batch_size > self._length:
                break
            self.end += self._batch_size
            self.batch.append(list(zip(*self._data[self.start:self.end])))
            self.start = self.end

def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible characters back into its (most likely) character representation."""
    return [id2char(c) for c in np.argmax(probabilities, 1)]

train_batches = BatchGenerator(train_list, batch_size)
valid_batches = BatchGenerator(valid_list, 1)

#print(train_batches.batch[0])
#sys.exit(1)

def logprob(predictions, labels):
    """Log-probability of the true labels in a predicted batch."""
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
    """Sample one element from a distribution assumed to be an array of normalized probabilities."""
    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distribution)):
        s += distribution[i]
        if s >= r:
            return i
    return len(distribution) - 1

def sample(prediction):
    """Turn a (column) prediction into 1-hot encoded samples."""
    p = np.zeros(shape=[1, vocab_size], dtype=np.float)
    p[0, sample_distribution(prediction[0])] = 1.0
    return p

def random_distribution():
    """Generate a random column of probabilities."""
    b = np.random.uniform(0.0, 1.0, size=[1, vocab_size])
    return b/np.sum(b, 1)[:,None]

num_nodes = 64

graph = tf.Graph()
with graph.as_default():
    # Parameters:
    # Input gate: input, previous output, and bias.
    ix = tf.Variable(tf.truncated_normal([vocab_size, num_nodes], -0.1, 0.1))
    im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ib = tf.Variable(tf.zeros([1, num_nodes]))
    # Forget gate: input, previous output, and bias.
    fx = tf.Variable(tf.truncated_normal([vocab_size, num_nodes], -0.1, 0.1))
    fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    fb = tf.Variable(tf.zeros([1, num_nodes]))
    # Memory cell: input, state and bias.                             
    cx = tf.Variable(tf.truncated_normal([vocab_size, num_nodes], -0.1, 0.1))
    cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    cb = tf.Variable(tf.zeros([1, num_nodes]))
    # Output gate: input, previous output, and bias.
    ox = tf.Variable(tf.truncated_normal([vocab_size, num_nodes], -0.1, 0.1))
    om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ob = tf.Variable(tf.zeros([1, num_nodes]))
    # Variables saving state across unrollings.
    saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    # Classifier weights and biases.
    w = tf.Variable(tf.truncated_normal([num_nodes, ch_vocab_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([ch_vocab_size]))

    # Definition of the cell computation.
    def lstm_cell(i, o, state):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf Note that in this formulation, we omit the various connections between the previous state and the gates."""
        input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
        forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
        update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
        state = forget_gate * state + input_gate * tf.tanh(update)
        output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
        return output_gate * tf.tanh(state), state

    # Input data.
    train_inputs = []
    train_labels = []

    for _ in range(padding_length):
        train_inputs.append(tf.placeholder(tf.float32, shape=[batch_size, vocab_size]))
        train_labels.append(tf.placeholder(tf.float32, shape=[batch_size, ch_vocab_size]))

    # Unrolled LSTM loop.
    outputs = list()
    output = saved_output
    state = saved_state
    for i in train_inputs:
        output, state = lstm_cell(i, output, state)
        outputs.append(output)

    # State saving across unrollings.
    with tf.control_dependencies([saved_output.assign(output),saved_state.assign(state)]):
        # Classifier.
        logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, train_labels)))

    # Optimizer.
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

    # Predictions.
    train_prediction = tf.nn.softmax(logits)
    
    # Sampling and validation eval: batch 1, no unrolling.
    #sample_input = tf.placeholder(tf.float32, shape=(1, vocabulary_size))
    #saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
    #saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
    #reset_sample_state = tf.group(saved_sample_output.assign(tf.zeros([1, num_nodes])),saved_sample_state.assign(tf.zeros([1, num_nodes])))
    #sample_output, sample_state = lstm_cell(sample_input, saved_sample_output, saved_sample_state)
    #with tf.control_dependencies([saved_sample_output.assign(sample_output),saved_sample_state.assign(sample_state)]):
    #    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

num_steps = 7001
summary_frequency = 100

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    #print('Initialized')
    mean_loss = 0
    for step in range(num_steps):
        for batches in train_batches.batch:
            feed_dict = {}
            for i in range(padding_length):
                feed_dict[train_inputs[i]] = batches[0][i]
                feed_dict[train_labels[i]] = batches[1][i]
            _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
            mean_loss += l
            if step % summary_frequency == 0:
                sys.exit(1)
                if step > 0:
                    mean_loss = mean_loss / summary_frequency
                # The mean loss is an estimate of the loss over the last few batches.
                print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
                mean_loss = 0
                labels = np.concatenate(list(batches)[1:])
                print('Minibatch perplexity: %.2f' % float(np.exp(logprob(predictions, labels))))
                if step % (summary_frequency * 10) == 0:
                    # Generate some samples.
                    print('=' * 80)
                    for _ in range(5):
                        feed = sample(random_distribution())
                        sentence = characters(feed)[0]
                        reset_sample_state.run()
                        for _ in range(79):
                            prediction = sample_prediction.eval({sample_input: feed})
                            feed = sample(prediction)
                            sentence += characters(feed)[0]
                        print(sentence)
                    print('=' * 80)
                # Measure validation set perplexity.
                reset_sample_state.run()
                valid_logprob = 0
                for _ in range(valid_size):
                    b = valid_batches.next()
                    predictions = sample_prediction.eval({sample_input: b[0]})
                    valid_logprob = valid_logprob + logprob(predictions, b[1])
                print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size)))
