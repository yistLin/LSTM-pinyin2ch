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
            batch = []
            for (i, line) in enumerate(f):
                line = line.strip().split('\t')
                source, target = line[0].strip().split(), line[1].strip().split()
                source_pad_len = self.seq_len - len(source)
                target_pad_len = self.seq_len - len(target)
                batch.append((
                                list(map(lambda x: self.__source_map.word2id[x], source)) + [source_PADID] * source_pad_len, 
                                list(map(lambda x: self.__target_map.word2id[x], target)) + [target_PADID] * target_pad_len
                            ))
                if len(batch) == self.batch_size:
                    yield list(zip(*batch))
                    batch.clear()

class Map():
    def __init__(self, words):
        self.__words = words
        self.id2word = None
        self.word2id = None
        self.size = 0
        self.PADID = None
        self.__build_map()

    def __build_map(self):
        self.__words += ['_PAD'] + ['_UNK']
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
            line = line.strip('\n').split('\t')
            source, target = line[0].strip(), line[1].strip()
            source_list += source.split()
            target_list += target.split()

    source_map = Map(source_list)
    target_map = Map(target_list)
    return source_map, target_map

def train(arg):
    source_map, target_map = preprocess(arg.train_data)
    train_batches = BatchGenerator(arg.train_data, arg.batch_size, arg.seq_len, source_map, target_map)
    # for batch in train_batches.next():
    #     print(batch)

    # seq2seqModel = Model(arg.rnn_size, arg.seq_len, arg.batch_size)
    seq2seqModel = Model(arg.rnn_size, arg.batch_size, arg.seq_len)
    seq2seqModel.build_graph()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range(100):
            sess.run(train_step, feed_dict={X:,Y:})
            result = sess.run(tf.argmax(logits, 1))
            print(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train seq2seq model.')
    parser.add_argument('--train_data', action='store', dest='train_data', required=True)
    parser.add_argument('--batch_size', action='store', dest='batch_size', type=int, required=True)
    parser.add_argument('--rnn_size', action='store', dest='rnn_size', type=int, required=True)
    parser.add_argument('--seq_len', action='store', dest='seq_len', type=int, required=True)
    train(parser.parse_args())
