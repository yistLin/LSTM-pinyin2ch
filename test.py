import sys
import argparse
import numpy as np
import tensorflow as tf

def test(arg):
    ckpt_dir = arg.ckpt_dir
    ckpt_step = arg.ckpt_step
    test_data = arg.test_data
    map_dir = arg.map_dir
    seq_len = arg.seq_len

    source_id2word = np.load('{}/source.npy'.format(map_dir)).tolist()
    target_id2word = np.load('{}/target.npy'.format(map_dir)).tolist()

    source_word2id = {y: x for (x, y) in enumerate(source_id2word)}
    target_word2id = {y: x for (x, y) in enumerate(target_id2word)}
    with tf.Session() as sess:
        seq2seqModel = tf.train.import_meta_graph('{}/model.ckpt-{}.meta'.format(ckpt_dir, ckpt_step))
        seq2seqModel.restore(sess, '{}/model.ckpt-{}'.format(ckpt_dir, ckpt_step))

        feed_previous = tf.get_collection('feed_previous')[0]
        output_keep_prob = tf.get_collection('output_keep_prob')[0]
        encode_inputs, decode_inputs, outputs = [], [], []
        for i in range(seq_len):
            encode_inputs.append(tf.get_collection('encode_{}'.format(i))[0])
            decode_inputs.append(tf.get_collection('decode_{}'.format(i))[0])
            outputs.append(tf.get_collection('output_{}'.format(i))[0])
        
        # print('feed:', feed_previous)
        # print('encodes:', encode_inputs)
        # print('decodes:', decode_inputs)
        # print('outputs:', outputs)

        if test_data is not None:
            print('[Test]: file {}'.format(test_data))
            with open(test_data) as f:
                cnt, acc = 0, 0
                for line in f:
                    line = line.strip().split('\t')
                    source, target = line[1].strip().split(), line[0].strip().split()
                    target = [target_word2id.get(x, target_word2id['_UNK']) for x in target]                    

                    pad_len = seq_len - len(source)
                    if pad_len < 0:
                        continue

                    _encode_inputs = list(reversed([source_word2id.get(x, source_word2id['_UNK']) for x in source] + [source_word2id['_PAD']] * pad_len))
                    _decode_inputs = [target_word2id['_GO']] + [target_word2id['_PAD']] * (seq_len - 1)
                    
                    feed_dict = {}
                    feed_dict[output_keep_prob] = 1.0
                    feed_dict[feed_previous] = True
                    for i in range(seq_len):
                        feed_dict[encode_inputs[i]] = [_encode_inputs[i]]
                        feed_dict[decode_inputs[i]] = [_decode_inputs[i]] 

                    _outputs = sess.run([outputs], feed_dict=feed_dict)
                    _outputs = [np.argmax(x) for x in _outputs[0]]
                    _outputs = _outputs[:len(target)]

                    correct = np.sum(np.equal(target, _outputs))
                    acc += correct / len(target)
                    cnt += 1

                    if cnt % 100 == 0:
                        print('Accuracy over {} sentences: {:.5f}%'.format(cnt, (acc / cnt)*100))

                print('Accuracy over {} sentences: {:.5f}%'.format(cnt, (acc / cnt)*100))
                    
        else:
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
                feed_dict[output_keep_prob] = 1.0
                feed_dict[feed_previous] = True
                for i in range(seq_len):
                    feed_dict[encode_inputs[i]] = [_encode_inputs[i]]
                    feed_dict[decode_inputs[i]] = [_decode_inputs[i]]

                _outputs = sess.run([outputs], feed_dict=feed_dict)
                _outputs = [target_id2word[np.argmax(x)] for x in _outputs[0]]
                if '_EOS' in _outputs:
                    _outputs = _outputs[:_outputs.index('_EOS')]
                print(' '.join(_outputs))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test seq2seq model.')
    parser.add_argument('--map_dir', action='store', dest='map_dir', default='map')
    parser.add_argument('--ckpt_dir', action='store', dest='ckpt_dir', required=True)
    parser.add_argument('--ckpt_step', action='store', dest='ckpt_step', required=True)
    parser.add_argument('--seq_len', action='store', dest='seq_len', type=int, required=True)
    parser.add_argument('--test_data', action='store', dest='test_data')
    test(parser.parse_args())
