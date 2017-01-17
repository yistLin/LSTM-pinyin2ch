import sys
import argparse
from model import *
from collections import Counter

class BatchGenerator():
    def __init__(self, filename, batch_size, seq_len, source_map, target_map):
        self.filename = filename
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.__source_map = source_map
        self.__target_map = target_map
 
    def next(self):
        source_PADID = self.__source_map.PADID
        target_PADID = self.__target_map.PADID
        with open(self.filename, 'r') as f:
            batch_source = []
            batch_target = []
            for (i, line) in enumerate(f):
                line = line.strip().split('\t')
                source, target = line[0].strip().split(), line[1].strip().split()
                source_pad_len = self.seq_len - len(source)
                target_pad_len = self.seq_len - len(target)
                source_id_list = list(map(lambda x: self.__source_map.word2id[x], source))
                target_id_list = list(map(lambda x: self.__target_map.word2id[x], target))
                batch_source.append(source_id_list + [source_PADID] * source_pad_len)
                batch_target.append(target_id_list + [target_PADID] * target_pad_len)
                if len(batch_source) == self.batch_size:
                    yield (list(zip(*batch_source)), list(zip(*batch_target)))
                    batch_source.clear()
                    batch_target.clear()

class Map():
    def __init__(self, words):
        self.__words = words
        self.id2word = None
        self.word2id = None
        self.size = 0
        self.PADID = None
        self.__build_map()

    def __build_map(self):
        self.__words += ['_PAD'] + ['_UNK'] + ['_GO'] + ['_EOS']
        counter = Counter(self.__words)
        self.id2word = sorted(counter, key=counter.get, reverse=True)
        self.word2id = {y: x for (x, y) in enumerate(self.id2word)}
        self.size = len(self.id2word)
        self.PADID = self.word2id['_PAD']

def preprocess(filename):
    print('[Preprocess]: file {}'.format(filename))
    source_list, target_list = [], []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            source, target = line[0].strip(), line[1].strip()
            source_list += source.split()
            target_list += target.split()
    
    source_map = Map(source_list)
    target_map = Map(target_list)
    return source_map, target_map
    
def train(arg):
    train_data = arg.train_data
    rnn_size = arg.rnn_size
    seq_len = arg.seq_len
    batch_size = arg.batch_size
    num_epoch = arg.num_epoch   
 
    source_map, target_map = preprocess(train_data)
    
    seq2seqModel = Model(rnn_size, seq_len, batch_size, source_map.size, target_map.size)
    seq2seqModel.build_graph()
    with tf.Session(graph=seq2seqModel.graph) as sess:
        sess.run(seq2seqModel.init_op)
        for epoch in range(arg.num_epoch):
            train_batches = BatchGenerator(train_data, batch_size, 
                    seq_len, source_map, target_map)
            feed_dict = {}
            for batch in train_batches.next():
                    for i in range(seq_len):
                        feed_dict[seq2seqModel.encode_inputs[i]] = batch[0][i]
                        feed_dict[seq2seqModel.decode_inputs[i]] = batch[1][i]
                        feed_dict[seq2seqModel.targets[i]] = batch[1][i]

                    _, loss = sess.run([seq2seqModel.optimizer, seq2seqModel.loss], feed_dict=feed_dict)
                    outputs = sess.run([seq2seqModel.outputs], feed_dict=feed_dict)
                    print('loss: {:.10f}'.format(loss))
                    print('targets: {}'.format(batch[1]))
                    print('outputs: {}'.format(outputs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train seq2seq model.')
    parser.add_argument('-train_data', action='store', dest='train_data', required=True)
    parser.add_argument('-batch_size', action='store', dest='batch_size', type=int, required=True)
    parser.add_argument('-rnn_size', action='store', dest='rnn_size', type=int, required= True)
    parser.add_argument('-seq_len', action='store', dest='seq_len', type=int, required=True)
    parser.add_argument('-num_epoch', action='store', dest='num_epoch', type=int, required=True)
    train(parser.parse_args())
