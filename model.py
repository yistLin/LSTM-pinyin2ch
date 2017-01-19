import tensorflow as tf

class Model():

    def __init__(self, rnn_size, seq_len, source_vocab_size, target_vocab_size, embedding_size):
        self.__rnn_size = rnn_size
        self.__seq_len = seq_len
        self.__source_vocab_size = source_vocab_size
        self.__target_vocab_size = target_vocab_size
        self.__embedding_size = embedding_size

    def build_graph(self):
        rnn_size = self.__rnn_size
        seq_len = self.__seq_len
        source_vocab_size = self.__source_vocab_size
        target_vocab_size = self.__target_vocab_size
        embedding_size = self.__embedding_size

        print('[Build graph]')
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.encode_inputs = [tf.placeholder(tf.int32, shape=(None,)) for _ in range(seq_len)]
            self.decode_inputs = [tf.placeholder(tf.int32, shape=(None,)) for _ in range(seq_len)]
            self.targets = [tf.placeholder(tf.int32, shape=(None,)) for _ in range(seq_len)]
            self.feed_previous = tf.placeholder(tf.bool)
            self.output_keep_prob = tf.placeholder(tf.float32)

            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
            self.__rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
                    rnn_cell,
                    output_keep_prob=self.output_keep_prob
                    )

            self.outputs, self.__state = tf.nn.seq2seq.embedding_rnn_seq2seq(
                    self.encode_inputs, 
                    self.decode_inputs, 
                    self.__rnn_cell,
                    source_vocab_size, 
                    target_vocab_size, 
                    embedding_size=embedding_size, 
                    feed_previous=self.feed_previous
                    )

            weights = [tf.fill(tf.shape(self.encode_inputs[0]), 1.0) for _ in range(seq_len)]
            self.loss = tf.nn.seq2seq.sequence_loss(self.outputs, self.targets, weights)

            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

            self.init_op = tf.global_variables_initializer()
