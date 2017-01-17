import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

class BasicRNN():

    def __init__(self, rnn_size, batch_size, seq_len):
        self.rnn_size = rnn_size
        self.batch_size = batch_size
        self.seq_len = seq_len

    def build_graph():
        # RNN Model
        X = tf.placeholder(tf.float32, shape=(self.batch_size, self.seq_len, self.rnn_size))
        Y = tf.placeholder(tf.float32, shape=(self.batch_size, self.seq_len))

        cell = rnn_cell.BasicRNNCell(self.rnn_size)
        state = tf.zeros([self.batch_size, cell.state_size])
        X_split = tf.split(0, time_step_size, X)

        outputs, states = tf.nn.seq2seq.rnn_decoder(X_split, state, cell)
        print(states)
        print(outputs)

        logits = tf.reshape(tf.concat(1, outputs), [-1, self.rnn_size])
        targets = tf.reshape(Y , [-1])
        weights = tf.ones([self.seq_len * self.batch_size])

        loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights])
        cross_entropy = tf.reduce_sum(loss) / self.batch_size
        train_step = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cross_entropy)

