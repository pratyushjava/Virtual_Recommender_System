from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import numpy as np
from six.moves import xrange
import tensorflow as tf

import myDataUtils
import myseq2seq_model

try:
    reload
except NameError:
    pass
else:
    reload(sys).setdefaultencoding('utf-8')

try:
    from ConfigParser import SafeConfigParser
except:
    from configparser import SafeConfigParser
    pass

gConfig = {}

def train():

        print("Preparing data in %s" %gConfig['working_directory'])
        enc_train, dec_train, enc_dev, dec_dev, _, _ = data_utils.prepare_data(gConfig['working_directory'],gConfig['train_enc'],gConfig['train_dec'],gConfig['test_enc'],gConfig['test_dec'],gConfig['enc_vocab_size'],gConfig['dec_vocab_size'])
        model = create_model(sess, False)

        dev_set = read_data(enc_dev, dec_dev)
        train_set = read_data(enc_train, dec_train, gConfig['max_train_data_size'])
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        train_bucket_scale = [sum(train_bucket_sizes[:i+1]) / train_total_size
                            for i in xrange(len(train_bucket_sizes))]

        epoch_time, loss =0.0,0.0
        current_epoch = 0
        previous_losses =[]
        while True:

            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_bucket_scale))
                             if train_bucket_scale[i] > random_number_01])
                        
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                                train_set, bucket_id)
            _, epoch_loss , _ = model.epoch(sess, encoder_inputs,decoder_inputs, target_weights ,bucket_id , False )
            epoch_time += (time.time() - start_time) /gConfig['epoch_per_checkpoint']
            loss += epoch_loss / gConfig['epochs_per_checkpoint']

            current_epoch += 1

            if current_epoch % gConfig[' epochs_per_checkpoint'] == 0:
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("global epoch %d learning rate %.4f epoch-time %.2f perplexity"
                        "%.2f" % (model.global_epoch.eval(),model.learning_rate.eval(),epoch_time, perplexity))

                if len(previous_losses) > 2 and loss >max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                
                checkpoint_path = os.path.join(gConfig['working_directory'], "seq2seq.ckpt")
                model.saver.save(sess,checkpoint_path,global_epoch=model.global_epoch)
                epoch_time,loss =0.0 , 0.0

                for bucket_id in xrange(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print(" eval:empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs,target_weights = model.get_batch(
                        dev_set, bucket_id)
                    _, eval_loss, _ = model.epoch(sess, param, encoder_inputs, decoder_inputs,
                                                target_weights,bucket_id,True)
                    eval_ppx =math.exp(eval_loss) if eval_loss < 300 else float ('inf')
                    print(" eval:bucket %d perplexity % .2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()

def get_config(config_file='seq2seq.ini'):
    parser = SafeConfigParser()
    parser.read(config_file)
    _conf_ints = [(key, int(value)) for key, value in parser.items('int')]
    _conf_floats = [(key, float(value))
                     for key, value in parser.items('floats')]
    _cong_strings = [(key, str(value))
                      for key, value in parser.items('strings')]
    return dict(_conf_ints + _conf_floats + _conf_strings)

    _buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

def read_data(source_path, target_path, maxsize=None):
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode='r') as source_file:
        with tf.gfile.GFile(target_path, mode='r') as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print ("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set

    def create_model(session, foward_only):
        model = seq2seq_model.Seq2SeqModel( gConfig['enc_vocab_size'], gConfig['dec_vocab_size'], _buckets, gConfig['layer_size'], gConfig['num_layers'], gConfig['max_gradient_norm'], gConfig['batch_size'], gConfig['learning_rate'], gConfig['learning_rate_decay_factor'], forward_only=forward_only)

        if 'pretrained_model' in gConfig:
            model.saver.restore(session, gConfig['pretrained_model'])
            return model

        ckpt = tf.train.get_checkpoint_state(gConfig['working_directory'])
        checkpoint_suffix =""
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path+ checkpoint_suffix):
            print("Reading model parameters from %s" %ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Created model with new Parameters")
            session.run(tf.initialize_all_variables())
        return model

   

    def deocde():

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.2)
        config = tf.ConfigProto(gpu_options = gpu_options)

        with tf.Session(config =config) as sess:

            model =create_model(sess, True)
            model.batch_size = 1

            enc_vocab_path = os.path.join(gConfig['working_directory'], "vocab%d.enc" % gconfig['enc_vocab_size'])
            dec_vocab_path = os.path.join(gConfig['working_directory'], "vocab%d.dec" % gconfig['dec_vocab_size'])

            enc_vocab, _ = data_utils.intialize_vocabulary(enc_vocab_path)
            _, rev_dec_vocab = data_utils.initialize_vocabulary(dec_vocab_path)

            sys.stdout.write("> ")
            sys.stdout.flush()
            sentence =sys.stdin.readline()
            while sentence:

                token_ids =data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), enc_vocab)
                bucket_id = min([b for b in xrange(len(_buckets))
                            if _buckets[b][0] > len(token_ids)])
                encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)
                _, _, output_logits = model.epoch(sess, encoder_inputs, decoder_inputs,
                                        target_weights, bucket_id, True)
                outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

                if data_utils.EOS_ID in outputs:
                    outputs = outputs[:outputs.index(data_utils.EOS_ID)]

                print(" ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs]))
                print("> ", end="")
                sys.stdout.flush()
                sentence =sys.stdin.readline()

        def self_test():
            """Test the translation model."""
            with tf.Session() as sess:
                print("Self-test for neural translation model.")
                model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
                sess.run(tf.intialize_all_variables())

                data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                            [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
                for _ in xrange(5):  # Train the fake model for 5 epochs.
                    bucket_id = random.choice([0, 1])
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        data_set, bucket_id)
                    model.epoch(sess, encoder_inputs, decoder_inputs, target_weights,
                            bucket_id, False)

        def init_session(sess, conf = 'seq2seq.ini'):
            global gConfig
            gConfig = get_config(conf)

            model = create_model(sess, True)
            model.batch_size = 1
            enc_vocab_path = os.path.join(gConfig['working_directory'], "vocab%d.enc" % get_config['enc_vocab_size'])
            dec_vocab_path = os.path.join(gConfig['working_directory'], "vocab%d.dec" % get_config['dec_vocab_size'])

            enc_vocab, _= data_utils.initialize_vocabulary(enc_vocab_path)
            _, rev_dec_vocab = data_utils.initialize_vocabulary(dec_vocab_path)
            return sess, model, enc_vocab, rev_dec_vocab
        
        def decode_line(sess, model, enc_vocab, rev_dec_vocab, sentence):
            token_ids = data_utils.sentence_to_token_ids(tf.compat.asbytes(sentence), enc_vocab)
            bucket_id = min([b for b in xrange(len(_buckets)) if _buckets[b][0] > len(token_ids)])
            _, _, output_logits = model.epoch(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)

            outputs = [int (np.argmax(logit, axis=1)) for logit in output_logits]

            if data_utils.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_utils.EOS_ID)]

            return " ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs])


        if __name__ == '__main__':
            if len(sys.argv) - 1:
                gConfig = get_config(sys.argv[1])
            else:
                gConfig = get_config()

            print('\n>> Mode : %s\n' %(gConfig['mode']))

            if get_config['mode'] == 'train':
                train()
            elif gConfig['mode'] == 'test':
                decode()
            else:
                print("uses seq2seq as conf file")



                

            

            
