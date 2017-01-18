import tensorflow as tf
import numpy as np

class BasicRNN():

    def __init__(self, rnn_size, batch_size, seq_len):
        self.rnn_size = rnn_size
        self.batch_size = batch_size
        self.seq_len = seq_len

    def build_graph(self):
        # char_rdic = ['h', 'e', 'l', 'o']
        # char_dic = {w: i for i, w in enumerate(char_rdic)}
        # sample = [char_dic[c] for c in 'hello']
        # x_data = np.array([
        #     [1,0,0,0],
        #     [0,1,0,0],
        #     [0,0,1,0],
        #     [0,0,1,0]],
        #     dtype=np.float32)
        # self.batch_size = 1
        # self.rnn_size = 4
        # self.seq_len = 4

        # RNN Model
        self.X = [tf.placeholder(tf.int32, shape=(self.batch_size)) for _ in range(self.seq_len)]
        self.Y = [tf.placeholder(tf.int32, shape=(self.batch_size)) for _ in range(self.seq_len)]

        X = list(map(lambda x: tf.one_hot(x, self.rnn_size, on_value=1.0, off_value=0.0, axis=-1), self.X))

        cell = tf.nn.rnn_cell.BasicRNNCell(self.rnn_size)
        state = tf.zeros([self.batch_size, cell.state_size])

        outputs, state = tf.nn.seq2seq.rnn_decoder(X, state, cell)

        logits = tf.reshape(tf.concat(1, outputs), [-1, self.rnn_size])
        targets = tf.reshape(self.Y, [-1])
        weights = tf.ones([self.seq_len * self.batch_size])
        
        outputs_trans = tf.transpose(outputs, perm=[1, 0, 2])
        # print('after reshape:', tf.reshape(tf.concat(0, outputs), []))
        # self.output = tf.arg_max(logits, 1)
        outputs_id = tf.arg_max(outputs_trans, 2)
        self.output = tf.split(0, self.batch_size, outputs_id)
        print('output:', self.output.get_shape())

        loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights])
        self.cross_entropy = tf.reduce_sum(loss) / self.batch_size
        self.train_step = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(self.cross_entropy)

