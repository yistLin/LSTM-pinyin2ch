import sys
import argparse
import numpy as np
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
        source_UNKID = self.__source_map.UNKID
        target_PADID = self.__target_map.PADID
        target_UNKID = self.__target_map.UNKID
        target_GOID  = self.__target_map.GOID
        target_EOSID = self.__target_map.EOSID
        with open(self.filename, 'r') as f:
            batch_encode = []
            batch_decode = []
            batch_target = []
            for (i, line) in enumerate(f):
                line = line.strip().split('\t')
                source, target = line[1].strip().split(), line[0].strip().split()

                source_pad_len = self.seq_len - len(source)
                target_pad_len = self.seq_len - len(target) - 1

                # length of sequence is larger than seq len
                if source_pad_len < 0 or target_pad_len < 0:
                    continue

                source_id_list = list(map(lambda x: self.__source_map.word2id.get(x, source_UNKID), source))
                target_id_list = list(map(lambda x: self.__target_map.word2id.get(x, target_UNKID), target))

                batch_encode.append(source_id_list + [source_PADID] * source_pad_len)
                batch_decode.append([target_GOID]  + target_id_list + [target_PADID] * target_pad_len)
                batch_target.append(target_id_list + [target_EOSID] + [target_PADID] * target_pad_len)

                if len(batch_encode) == self.batch_size:
                    yield   {
                                'encode': list(reversed(list(zip(*batch_encode)))),
                                'decode': list(zip(*batch_decode)),
                                'target': list(zip(*batch_target))
                            }
                    batch_encode.clear()
                    batch_decode.clear()
                    batch_target.clear()

class Map():
    def __init__(self, words):
        self.__words = words
        self.id2word = None
        self.word2id = None
        self.size = 0
        self.__build_map()

    def __build_map(self):
        self.__words += ['_PAD'] + ['_UNK'] + ['_GO'] + ['_EOS']
        counter = Counter(self.__words)
        self.id2word = sorted(counter, key=counter.get, reverse=True)
        self.word2id = {y: x for (x, y) in enumerate(self.id2word)}
        self.size = len(self.id2word)
        self.PADID = self.word2id['_PAD']
        self.UNKID = self.word2id['_UNK']
        self.GOID  = self.word2id['_GO']
        self.EOSID = self.word2id['_EOS']

    def save(self, filename):
        print('[Save]: save map at {}'.format(filename))
        np.save(filename, self.id2word)

def preprocess(filename):
    print('[Preprocess]: file {}'.format(filename))
    source_list, target_list = [], []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip('\n').split('\t')
            source, target = line[1].strip(), line[0].strip()
            source_list += source.split()
            target_list += target.split()

    source_map = Map(source_list)
    target_map = Map(target_list)
    return source_map, target_map

def train(arg):
    train_data = arg.train_data 
    valid_data = arg.valid_data
    rnn_size = arg.rnn_size
    seq_len = arg.seq_len
    batch_size = arg.batch_size
    num_epoch = arg.num_epoch
    step_print = arg.step_print
    save_num_epoch = arg.save_num_epoch
    save_path = arg.save_path
    map_dir = arg.map_dir
    embedding_size = arg.embedding_size

    source_map, target_map = preprocess(train_data)
    source_map.save('{}/source'.format(map_dir))
    target_map.save('{}/target'.format(map_dir))

    seq2seqModel = Model(rnn_size, seq_len, source_map.size, target_map.size, embedding_size)
    seq2seqModel.build_graph()

    loss = step_cnt = 0

    print('[Training]')
    print('\ttrain data: {}'.format(train_data))
    if valid_data is not None:
        print('\tvalid data: {}'.format(valid_data))

    with tf.Session(graph=seq2seqModel.graph) as sess:	
        # Create saver for model
        saver = tf.train.Saver()

        tf.add_to_collection('feed_previous', seq2seqModel.feed_previous)
        for i in range(seq_len):
            tf.add_to_collection('encode_{}'.format(i), seq2seqModel.encode_inputs[i])
            tf.add_to_collection('decode_{}'.format(i), seq2seqModel.decode_inputs[i])
            tf.add_to_collection('output_{}'.format(i), seq2seqModel.outputs[i])
            
        model_save_path = save_path + '/model.ckpt'
        print('[Model]: model save at {}'.format(model_save_path))

        sess.run(seq2seqModel.init_op)
        for epoch in range(arg.num_epoch):
            train_batches = BatchGenerator(train_data, batch_size,
                    seq_len, source_map, target_map)
            feed_dict = {}
            feed_dict[seq2seqModel.feed_previous] = False
            for batch in train_batches.next():
                for i in range(seq_len):
                    feed_dict[seq2seqModel.encode_inputs[i]] = batch['encode'][i]
                    feed_dict[seq2seqModel.decode_inputs[i]] = batch['decode'][i]
                    feed_dict[seq2seqModel.targets[i]]       = batch['target'][i]

                _, _loss = sess.run([seq2seqModel.optimizer, seq2seqModel.loss], feed_dict=feed_dict)
                loss += _loss
                step_cnt += 1
                if step_cnt % step_print == 0:
                    outputs = sess.run(seq2seqModel.outputs, feed_dict=feed_dict)
                    answer, predict = [], []
                    rand = np.random.randint(batch_size, size=1, dtype=int)
                    for (i, words) in enumerate(outputs):
                        answer.append(target_map.id2word[batch['target'][i][rand]])
                        predict.append(target_map.id2word[np.argmax(words, axis=1)[rand]])

                    print('[Epoch {}]: #batch {}, average loss: {:.6f}'.format(epoch, step_cnt, loss / step_print))
                    print('\tRandom sentence in train batch:')
                    print('\t\tanswer : {}'.format(' '.join(answer)))
                    print('\t\tpredict: {}'.format(' '.join(predict)))
                    print('')
                    loss = 0

                if (valid_data is not None) and (step_cnt % (step_print * 10) == 0):
                    valid_batches = BatchGenerator(valid_data, batch_size,
                            seq_len, source_map, target_map)
                    v_loss = v_step = 0
                    feed_dict = {}
                    feed_dict[seq2seqModel.feed_previous] = False
                    for v_batch in valid_batches.next():
                        for i in range(seq_len):
                            feed_dict[seq2seqModel.encode_inputs[i]] = v_batch['encode'][i]
                            feed_dict[seq2seqModel.decode_inputs[i]] = v_batch['decode'][i]
                            feed_dict[seq2seqModel.targets[i]]       = v_batch['target'][i]

                        _loss = sess.run(seq2seqModel.loss, feed_dict=feed_dict)
                        v_loss += _loss
                        v_step += 1
                    
                    print('[Valid data]: average loss: {:.6f}'.format(v_loss / v_step))
                    print('')
        
            # save the model
            if epoch % save_num_epoch == 0:
                path = saver.save(sess, model_save_path, global_step=epoch)
                print('\n[Train] save model to path: {}'.format(path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train seq2seq model.')
    parser.add_argument('--train_data', action='store', dest='train_data', required=True)
    parser.add_argument('--valid_data', action='store', dest='valid_data')
    parser.add_argument('--map_dir', action='store', dest='map_dir', default='map')
    parser.add_argument('--batch_size', action='store', dest='batch_size', type=int, required=True)
    parser.add_argument('--rnn_size', action='store', dest='rnn_size', type=int, required=True)
    parser.add_argument('--seq_len', action='store', dest='seq_len', type=int, required=True)
    parser.add_argument('--num_epoch', action='store', dest='num_epoch', type=int, required=True)
    parser.add_argument('--step_print', action='store', dest='step_print', type=int, default=100)
    parser.add_argument('--save_num_epoch', action='store', dest='save_num_epoch', type=int, default=1)
    parser.add_argument('--save_path', action='store', dest='save_path', required=True)
    parser.add_argument('--embedding_size', action='store', dest='embedding_size', type=int, default=256)
    train(parser.parse_args())
