import tensorflow as tf
import numpy as np
import sys
import random

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
min_length, padding_length = 3, 15

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
    if min_length <= len(raw_pinyin_list) <= padding_length:
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
    if min_length <= len(raw_pinyin_list) <= padding_length:
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

batch_size=32

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
                                      shape=[batch_size, padding_length, self.input_size],
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


# Function to convert batch input data to use scan ops of tensorflow.
def process_batch_input_for_RNN(batch_input):
    """
    Process tensor of size [5,3,2] to [3,5,2]
    """
    batch_input_ = tf.transpose(batch_input, perm=[2, 0, 1])
    X = tf.transpose(batch_input_)

    return X

# # Placeholder and initializers

hidden_layer_size = 64

y = tf.placeholder(tf.float32, shape=[batch_size, padding_length, ch_vocab_size], name='inputs')

# # Models

# Initializing rnn object
rnn = LSTM_cell(vocab_size, hidden_layer_size, ch_vocab_size)

# Getting all outputs from rnn
outputs = tf.transpose(rnn.get_outputs(), perm=[1, 0, 2])
print('outputs shape: ', outputs.get_shape())

# Getting final output through indexing after reversing
# last_output = tf.reverse(outputs, [True, False, False])[0, :, :]
# print('last output shape: ', last_output.get_shape())

# As rnn model output the final layer through Relu activation softmax is
# used for final output.
output = tf.nn.softmax(outputs, dim=-1)
print('output shape: ', output.get_shape())

# Computing the Cross Entropy loss
cross_entropy = -tf.reduce_sum(y * tf.log(output))

# Trainning with Adadelta Optimizer
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

# Calculatio of correct prediction and accuracy
correct_prediction = tf.equal(tf.argmax(y, axis=2), tf.argmax(output, axis=2))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100

# # Dataset Preparation
train_batches = BatchGenerator(train_list, batch_size)
valid_batches = BatchGenerator(valid_list, 1)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Iterations to do trainning
for epoch in range(300):

    total_loss, train_acc, test_acc = 0, 0, 0
    batches = train_batches.batch
    random.shuffle(batches)
    for batch in batches:
        _, loss, _train_acc = sess.run([train_step, cross_entropy, accuracy], feed_dict={rnn._inputs: batch[0], y: batch[1]})
        total_loss += loss / batch_size
        train_acc += _train_acc

        #Loss = sess.run(cross_entropy, feed_dict={
        #    rnn._inputs: train_batches.batch[0][0], y: train_batches.batch[0][1]})
        #Train_accuracy = str(sess.run(accuracy, feed_dict={
        #    rnn._inputs: train_batches.batch[0][0], y: train_batches.batch[0][1]}))
        Test_accuracy = str(sess.run(accuracy, feed_dict={
            rnn._inputs: train_batches.batch[0][0], y: train_batches.batch[0][1]}))

    sys.stdout.flush()
    print("\rIteration: %s Loss: %s Train Accuracy: %s%% Test Accuracy: %s%%" %
          (epoch, str(total_loss/len(train_batches.batch)), str(train_acc/len(train_batches.batch)), str(Test_accuracy))),
    sys.stdout.flush()
