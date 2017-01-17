import tensorflow as tf

class Model():

    def __init__(self, rnn_size, seq_len, batch_size, source_vocab_size, target_vocab_size):
        self.__rnn_size = rnn_size
        self.__seq_len = seq_len
        self.__batch_size = batch_size
        self.__source_vocab_size = source_vocab_size
        self.__target_vocab_size = target_vocab_size

    def build_graph(self):
        print('[Build graph]')
        rnn_size = self.__rnn_size
        seq_len = self.__seq_len
        batch_size = self.__batch_size
        source_vocab_size = self.__source_vocab_size
        target_vocab_size = self.__target_vocab_size
        print('\trnn size: {}'.format(rnn_size))
        print('\tseq len: {}'.format(seq_len))
        print('\tbatch_size: {}'.format(batch_size))
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.encode_inputs = [tf.placeholder(tf.int32, shape=(batch_size,)) for _ in range(seq_len)]
            self.decode_inputs = [tf.placeholder(tf.int32, shape=(batch_size,)) for _ in range(seq_len)]
            self.targets = [tf.placeholder(tf.int32, shape=(batch_size,)) for _ in range(seq_len)]
            #self.encode_inputs = tf.placeholder(tf.int32, shape=[batch_size, seq_len])
            #self.decode_inputs = tf.placeholder(tf.int32, shape=[batch_size, seq_len])
            #self.labels = tf.placeholder(tf.float32, shape=[batch_size, seq_len])

            #print('[Placeholder]')
            #print('\tencode inputs shape: {}'.format(self.encode_inputs.get_shape()))
            #print('\tdecode inputs shape: {}'.format(self.decode_inputs.get_shape()))
            #print('\tlabels shape: {}'.format(self.labels.get_shape()))

            self.__rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
            print('[RNN cell]')
            print('\toutput size: {}'.format(self.__rnn_cell.output_size))
            print('\tstate size: {}'.format(self.__rnn_cell.state_size))

            #self.split_inputs = tf.split(1, seq_len, self.encode_inputs)
            #print('[Split input]')
            #print('\tsplit input: {}'.format(self.split_inputs))
            # self.__state = tf.zeros([batch_size, self.__rnn_cell.state_size])

            self.outputs, self.__state = tf.nn.seq2seq.embedding_rnn_seq2seq(
                    self.encode_inputs, self.decode_inputs, self.__rnn_cell,
                    source_vocab_size, target_vocab_size, 256)
            print('[Output]')
            print('\toutputs: ', self.outputs)

            weights = [tf.constant(1.0, shape=[batch_size]) for _ in range(seq_len)]
            self.loss = tf.nn.seq2seq.sequence_loss(self.outputs, self.targets, weights)

            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

            self.init_op = tf.global_variables_initializer()
