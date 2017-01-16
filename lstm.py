#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import sys
import random

if len(sys.argv) < 8:
  print("Usage: ./lstm.py [#layer] [#batch] [train dir] [source vocabularies list] [train data] [valid data] [test data] [--decode]")
  sys.exit(-1)

print('\n********** Program Start **********')

# get train data and valid data filename
try:
    hidden_layer_size = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    train_dir = sys.argv[3]
    vocab_filename = sys.argv[4]
    train_data_filename = sys.argv[5]
    valid_data_filename = sys.argv[6]
    test_data_filename = sys.argv[7]
except Exception:
    print('[Error] cannot get data')
    sys.exit(1)

decode_mode = True if '--decode' in sys.argv else False

# open training file
train_data_f = open(train_data_filename, 'r')
valid_data_f = open(valid_data_filename, 'r')
vocab_f = open(vocab_filename, 'r')

# all pinyin and 0-9
py2id = {}
cnt = 0
for line in vocab_f:
    v = line.strip()
    py2id[v] = cnt
    cnt += 1

vocab_size = len(py2id)

ch2id = {}
id2ch = {}
ch_index = 2
train_list = []
ch2id['_PAD'] = 0
ch2id['_UNK'] = 1
id2ch[0] = '_PAD'
id2ch[1] = '_UNK'

min_padding_len, max_padding_len = 3, 15

# build chinese characters mapping
for line in train_data_f:
    line_sp = line.strip('\n').split('\t')
    raw_pinyin_list = line_sp[1].split()
    if min_padding_len <= len(raw_pinyin_list) <= max_padding_len:
        raw_ch_list = line_sp[0].split()
        for ch in raw_ch_list:
            if ch not in ch2id:
                ch2id[ch] = ch_index
                id2ch[ch_index] = ch
                ch_index += 1

ch_vocab_size = len(ch2id)
train_data_f.seek(0)

print('\n[Dict] py2id, ch2id and id2ch built')

# train data and valid data
# line_sp[0] is chinese characters as label
# line_sp[1] is pinyin characters as input

# read in train data
for line in train_data_f:
    line_sp = line.strip('\n').split('\t')
    raw_pinyin_list = line_sp[1].split()
    raw_ch_list = line_sp[0].split()
    if min_padding_len <= len(raw_pinyin_list) <= max_padding_len:
        raw_pinyin_list += ['_PAD'] * (max_padding_len - len(raw_pinyin_list))
        raw_ch_list += ['_PAD'] * (max_padding_len - len(raw_ch_list))
        train_list.append( (raw_pinyin_list, raw_ch_list) )

# read in valid data
valid_list = []
for line in valid_data_f:
    line_sp = line.strip('\n').split('\t')
    raw_pinyin_list = line_sp[1].split()
    raw_ch_list = line_sp[0].split()
    if min_padding_len <= len(raw_pinyin_list) <= max_padding_len:
        raw_pinyin_list += ['_PAD'] * (max_padding_len - len(raw_pinyin_list))
        raw_ch_list += ['_PAD'] * (max_padding_len - len(raw_ch_list))
        valid_list.append( (raw_pinyin_list, raw_ch_list) )

print('\n[Data] train and valid data read')

# close training file
vocab_f.close()
train_data_f.close()
valid_data_f.close()

# batch_size = 32

class BatchGenerator(object):
    def __init__(self, data, batch_size):
        self._data = data
        self._batch_size = batch_size
        self.start = 0
        self.end = 0
        self._length = len(data)

    def next(self):
        while True:
            if self.end + self._batch_size > self._length:
                break
            self.end += self._batch_size

            yield self.__get_one_hot()
            self.start = self.end

    def __get_one_hot(self):
        batch_list = []
        for batch in self._data[self.start:self.end]:
            pin_list = np.zeros(shape=(max_padding_len, vocab_size), dtype=np.float)
            ch_list = np.zeros(shape=(max_padding_len, ch_vocab_size), dtype=np.float)
            for (i, word) in enumerate(batch[0]):
                w = py2id[word] if word in py2id else py2id['_UNK']
                pin_list[i, w] = 1.0
            for (i, word) in enumerate(batch[1]):
                w = ch2id[word] if word in ch2id else ch2id['_UNK']
                ch_list[i, w] = 1.0
            batch_list.append( [pin_list, ch_list] )
        return list(zip(*batch_list))

class LSTM_cell(object):
    """
    LSTM cell object which takes 3 arguments for initialization.
    input_size = Input Vector size
    hidden_layer_size = Hidden layer size
    target_size = Output vector size
    """
    def __init__(self, input_size, hidden_layer_size, target_size):

        # Initialization of given values
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.target_size = target_size

        # Weights and Bias for input and hidden tensor
        self.Wi = tf.Variable(tf.zeros(
            [self.input_size, self.hidden_layer_size]))
        self.Ui = tf.Variable(tf.zeros(
            [self.hidden_layer_size, self.hidden_layer_size]))
        self.bi = tf.Variable(tf.zeros([self.hidden_layer_size]))

        self.Wf = tf.Variable(tf.zeros(
            [self.input_size, self.hidden_layer_size]))
        self.Uf = tf.Variable(tf.zeros(
            [self.hidden_layer_size, self.hidden_layer_size]))
        self.bf = tf.Variable(tf.zeros([self.hidden_layer_size]))

        self.Wog = tf.Variable(tf.zeros(
            [self.input_size, self.hidden_layer_size]))
        self.Uog = tf.Variable(tf.zeros(
            [self.hidden_layer_size, self.hidden_layer_size]))
        self.bog = tf.Variable(tf.zeros([self.hidden_layer_size]))

        self.Wc = tf.Variable(tf.zeros(
            [self.input_size, self.hidden_layer_size]))
        self.Uc = tf.Variable(tf.zeros(
            [self.hidden_layer_size, self.hidden_layer_size]))
        self.bc = tf.Variable(tf.zeros([self.hidden_layer_size]))

        # Weights for output layers
        self.Wo = tf.Variable(tf.truncated_normal(
            [self.hidden_layer_size, self.target_size], mean=0, stddev=.01))
        self.bo = tf.Variable(tf.truncated_normal(
            [self.target_size], mean=0, stddev=.01))

        # Placeholder for input vector with shape[batch, seq, embeddings]
        self._inputs = tf.placeholder(tf.float32,
                                      shape=[None, max_padding_len, self.input_size],
                                      name='inputs')

        # Processing inputs to work with scan function
        self.processed_input = process_batch_input_for_RNN(self._inputs)

        '''
        Initial hidden state's shape is [1,self.hidden_layer_size]
        In First time stamp, we are doing dot product with weights to
        get the shape of [batch_size, self.hidden_layer_size].
        For this dot product tensorflow use broadcasting. But during
        Back propagation a low level error occurs.
        So to solve the problem it was needed to initialize initial
        hiddden state of size [batch_size, self.hidden_layer_size].
        So here is a little hack !!!! Getting the same shaped
        initial hidden state of zeros.
        '''

        self.initial_hidden = self._inputs[:, 0, :]
        self.initial_hidden = tf.matmul(
            self.initial_hidden, tf.zeros([input_size, hidden_layer_size]))

        self.initial_hidden = tf.pack(
            [self.initial_hidden, self.initial_hidden])

    # Function for LSTM cell.
    def Lstm(self, previous_hidden_memory_tuple, x):
        """
        This function takes previous hidden state and memory
         tuple with input and
        outputs current hidden state.
        """
        previous_hidden_state, c_prev = tf.unpack(previous_hidden_memory_tuple)

        # Input Gate
        i = tf.sigmoid(
            tf.matmul(x, self.Wi) +
            tf.matmul(previous_hidden_state, self.Ui) + self.bi
        )

        # Forget Gate
        f = tf.sigmoid(
            tf.matmul(x, self.Wf) +
            tf.matmul(previous_hidden_state, self.Uf) + self.bf
        )

        # Output Gate
        o = tf.sigmoid(
            tf.matmul(x, self.Wog) +
            tf.matmul(previous_hidden_state, self.Uog) + self.bog
        )

        # New Memory Cell
        c_ = tf.nn.tanh(
            tf.matmul(x, self.Wc) +
            tf.matmul(previous_hidden_state, self.Uc) + self.bc
        )

        # Final Memory cell
        c = f * c_prev + i * c_

        # Current Hidden state
        current_hidden_state = o * tf.nn.tanh(c)
        return tf.pack([current_hidden_state, c])

    # Function for getting all hidden state.
    def get_states(self):
        """
        Iterates through time/ sequence to get all hidden state
        """
        # Getting all hidden state throuh time
        all_hidden_states = tf.scan(self.Lstm,
                                    self.processed_input,
                                    initializer=self.initial_hidden,
                                    name='states')
        all_hidden_states = all_hidden_states[:, 0, :, :]
        return all_hidden_states

    # Function to get output from a hidden layer
    def get_output(self, hidden_state):
        """
        This function takes hidden state and returns output
        """
        output = tf.nn.relu(tf.matmul(hidden_state, self.Wo) + self.bo)
        return output

    # Function for getting all output layers
    def get_outputs(self):
        """
        Iterating through hidden states to get outputs for all timestamp
        """
        all_hidden_states = self.get_states()
        all_outputs = tf.map_fn(self.get_output, all_hidden_states)
        return all_outputs

print('\n[Model] Batch Generator and LSTM defined')

# Function to convert batch input data to use scan ops of tensorflow.
def process_batch_input_for_RNN(batch_input):
    """
    Process tensor of size [5,3,2] to [3,5,2]
    """
    batch_input_ = tf.transpose(batch_input, perm=[2, 0, 1])
    X = tf.transpose(batch_input_)
    return X

# # Placeholder and initializers

# hidden_layer_size = 32

y = tf.placeholder(tf.float32, shape=[None, max_padding_len, ch_vocab_size], name='inputs')

# # Models

# Initializing rnn object
rnn = LSTM_cell(vocab_size, hidden_layer_size, ch_vocab_size)

# Getting all outputs from rnn
outputs = tf.transpose(rnn.get_outputs(), perm=[1, 0, 2])

# As rnn model output the final layer through Relu activation softmax is
# used for final output.
output = tf.nn.softmax(outputs, dim=-1)
prediction = tf.argmax(output, axis=2)
label = tf.argmax(y, axis=2)

# Computing the Cross Entropy loss
cross_entropy = -tf.reduce_sum(y * tf.log(output))

# Trainning with Adadelta Optimizer
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

# Calculate ratio of correct prediction and accuracy
correct_prediction = tf.equal(label, prediction)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100

# Print out shape for checking
print('outputs:', outputs.get_shape())
print('output:', output.get_shape())
print('prediection:', prediction.get_shape())
print('label:', label.get_shape())
print('cross_entropy:', cross_entropy.get_shape())
print('correct_prediction:', correct_prediction.get_shape())
print('accuracy:', accuracy.get_shape())

print('\n[Train] create session')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Create saver for all variables
saver = tf.train.Saver()
model_save_path = train_dir + 'model.ckpt'

if decode_mode:
    ckpt = tf.train.get_checkpoint_state(train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('\n[Decode] model successfully restore from path: %s' % model_save_path)
    else:
        print('\n[Decode] fail to find checkpoint')
    sys.exit(1)

# Iterations to do trainning
total_loss, train_acc, test_acc = 0, 0, 0
batch_cnt, batch_print = 0, 500
max_epoch = 300

print('\n[Train] start to train')
print('[Train] epoch at most %d' % max_epoch)

for epoch in range(max_epoch):

    print('\n[Train] epoch %d' % epoch)

    # batch_cnt, total_loss, train_acc, test_acc = 0, 0, 0, 0
    train_batches = BatchGenerator(train_list, batch_size)
    for batch in train_batches.next():
        _, loss, _train_acc = sess.run([train_step, cross_entropy, accuracy], feed_dict={rnn._inputs: batch[0], y: batch[1]})
        total_loss += loss
        train_acc += _train_acc

        batch_cnt += 1
        if batch_cnt == batch_print:
            batch_pack = []
            valid_batch_size = 1

            valid_batches = BatchGenerator(valid_list, valid_batch_size)
            for vb in valid_batches.next():
                batch_pack.append( (vb[0][0], vb[1][0]) )
            batch_zip = list(zip(*batch_pack))
            valid_acc = sess.run(accuracy, feed_dict={
                    rnn._inputs: batch_zip[0],
                    y: batch_zip[1]})

            print("\nEpoch [%s] #batch: %s, loss: %s, train accuracy: %s%%, valid accuracy: %s%%" % (
                    epoch, str(batch_cnt),
                    str(total_loss/(batch_print*batch_size)),
                    str(train_acc/batch_print),
                    str(valid_acc) ) )
            sys.stdout.flush()
            batch_cnt = total_loss = train_acc = test_acc = 0

    # save the model
    if epoch % 5 == 0 and epoch != 0:
        save_path = saver.save(sess, model_save_path, global_step=epoch)
        print('\n[Train] save model to path: %s' % save_path)

