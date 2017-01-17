import tensorflow as tf

class Model():

    def __init__(self, rnn_size, seq_len, batch_size):
        self.__rnn_size = rnn_size
        self.__seq_len = seq_len
        self.__batch_size = batch_size

    def build_graph(self):
        print('[Build graph]')
        rnn_size = self.__rnn_size
        seq_len = self.__seq_len
        batch_size = self.__batch_size
        print('\trnn size: {}'.format(rnn_size))
        print('\tseq len: {}'.format(seq_len))
        print('\tbatch_size: {}'.format(batch_size))
        self.__graph = tf.Graph()
        with self.__graph.as_default():
            print('[RNN cell]: create')
            self.__rnn_cell = rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
            print('[State]: create')
            self.__state = state = tf.zeros([batch_size, rnn_cell.state_size])
