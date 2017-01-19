import sys
import argparse
import numpy as np
import tensorflow as tf

def test(arg):
    ckpt_dir = arg.ckpt_dir
    test_data = arg.test_data
    map_dir = arg.map_dir
    seq_len = arg.seq_len

    source_id2word = np.load('{}/source.npy'.format(map_dir)).tolist()
    target_id2word = np.load('{}/target.npy'.format(map_dir)).tolist()

    source_word2id = {y: x for (x, y) in enumerate(source_id2word)}
    target_word2id = {y: x for (x, y) in enumerate(target_id2word)}
    with tf.Session() as sess:
        seq2seqModel = tf.train.import_meta_graph('{}/model.ckpt-1.meta'.format(ckpt_dir))
        seq2seqModel.restore(sess, ckpt_dir + '/model.ckpt-1')

        feed_previous = tf.get_collection('feed_previous')[0]
        encode_inputs, decode_inputs, outputs = [], [], []
        for i in range(seq_len):
            encode_inputs.append(tf.get_collection('encode_{}'.format(i))[0])
            decode_inputs.append(tf.get_collection('decode_{}'.format(i))[0])
            outputs.append(tf.get_collection('output_{}'.format(i))[0])
        
        # print('feed:', feed_previous)
        # print('encodes:', encode_inputs)
        # print('decodes:', decode_inputs)
        # print('outputs:', outputs)

        while True:
            sys.stdout.write('> ')
            sys.stdout.flush()  
            sen = sys.stdin.readline()
            if not sen:
                break
            sen = sen.strip().split()
            pad_len = seq_len - len(sen)
            if pad_len < 0:
                print('length of sentence should be <= {}'.format(seq_len))
                continue

            _encode_inputs = list(reversed([source_word2id.get(x, source_word2id['_UNK']) for x in sen] + [source_word2id['_PAD']] * pad_len))
            _decode_inputs = [target_word2id['_GO']] + [target_word2id['_PAD']] * (seq_len - 1)

            feed_dict = {}
            feed_dict[feed_previous] = True
            for i in range(seq_len):
                feed_dict[encode_inputs[i]] = [_encode_inputs[i]]
                feed_dict[decode_inputs[i]] = [_decode_inputs[i]]

            _outputs = sess.run([outputs], feed_dict=feed_dict)
            _outputs = [target_id2word[np.argmax(x)] for x in _outputs[0]]
            print(' '.join(_outputs[:_outputs.index('_EOS')]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test seq2seq model.')
    parser.add_argument('--ckpt_dir', action='store', dest='ckpt_dir', required=True)
    parser.add_argument('--map_dir', action='store', dest='map_dir', default='map')
    parser.add_argument('--test_data', action='store', dest='test_data')
    parser.add_argument('--seq_len', action='store', dest='seq_len', type=int, required=True)
    test(parser.parse_args())
