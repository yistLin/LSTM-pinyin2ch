import tensorflow as tf
import numpy as np

class BasicRNN():

    def __init__(self, rnn_size, batch_size, seq_len):
        self.rnn_size = rnn_size
        self.batch_size = batch_size
        self.seq_len = seq_len

    def build_graph(self):
        # RNN Model
        self.X = [tf.placeholder(tf.int32, shape=(self.batch_size)) for _ in range(self.seq_len)]
        self.Y = [tf.placeholder(tf.int32, shape=(self.batch_size)) for _ in range(self.seq_len)]

        X = list(map(lambda x: tf.one_hot(x, self.rnn_size, on_value=1.0, off_value=0.0, axis=-1), self.X))

        cell = tf.nn.rnn_cell.BasicRNNCell(self.rnn_size)
        state = tf.zeros([self.batch_size, cell.state_size])

        self.outputs, state = tf.nn.seq2seq.rnn_decoder(X, state, cell)

        weights = [tf.ones([self.batch_size])] * self.seq_len

        outputs_trans = tf.transpose(self.outputs, perm=[1, 0, 2])
        self.output = tf.arg_max(outputs_trans, 2)

        self.loss = tf.nn.seq2seq.sequence_loss(self.outputs, self.Y, weights)
        self.train_step = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(self.loss)

