from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import sys

from six.moves import xrange
import tensorflow as tf

import myDataUtils

class Seq2SeqModel():

    def __init__(self,source_vocab_size, target_vocab_size,buckets, size,
                num_layers, max_gradient_norm , batch_size, learning_rate,
                learning_rate_deacy_op, use_lstm=False):

                self.source_vocab_size = source_vocab_size
                self.target_vocab_size =target_vocab_size
                self.buckets = buckets
                self.batch_size = batch_size
                self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
                self.learning_rate_deacy_op = self.learning_rate.assign(self.learning_rate * learning_rate_deacy_factor)
                self.global_epoch =tf.Variable(0,trainable=False)

                output_projection = None
                softmax_loss_function= None

                if num_samples > 0 and num_samples<self.target_vocab_size:
                    w = tf.get_variable("proj_w", [size, self.target_vocab_size])
                    w_t = tf.transpose(w)
                    b = tf.get_variable("proj_b", [self.target_vocab_size])
                    output_projection = (w,b)

                    def sampled_loss(inputs,labels):
                        labels = tf.reshape(labels, [1,-1])
                        return tf.nn.sampled_softmax_loss(w_t,b,inputs,labels,num_samples,self.target_vocab_size)
                    softmax_loss_function =sampled_loss

                single_cell = tf.nn.rnn_cell.GRUCell(size)
                if use_lstm:
                    single_cell = tf.nn.rnn.cell.BasicLSTMCell(size)
                cell = single_cell
                if num_layers > 1:
                    cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

                def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
                    return tf.nn.seq2seq.embedding_attention_seq2seq(
                            encoder_inputs, decoder_inputs,cell,
                            num_encoder_symbols = source_vocab_size,
                            num_decoder_symbols = target_vocab_size,
                            embedding_size = size,
                            output_projection = output_projection,
                            feed_previous =do_decode)

                    self.encoder_inputs = []
                    self.decoder_inputs = []
                    self.target_weights =[]
                    for i in xrange(buckets[-1][0]):
                        self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                                    name= "encoder{0}".formate(i)))
                    for i in xrange(buckets[-1][1] + 1):
                        self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                                name="decoder{0}".format(i)))
                        self.target_weights.append(tf.placeholder(tf.float32,shape=[None],
                                                                name= "weight{0}".format(i)))
                    target= [self.decoder_inputs[i+1]
                                for i in xrange(len(self.decoder_inputs) -1)]

                    if forward_only:
                        self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                            self.encoder_inputs, self.decoder_inputs, targets,
                            self.target_weights, buckets, lambda x, y: seq2seq_f(x,y,True),
                            softmax_loss_function = softmax_loss_function
                        )
                        if output_projection is not None:
                            for b in xrange(len(buckets)):
                                self.outputs[b] = [
                                    tf.matmul(output, output_projection[0] + output_projection[1])
                                    for output in self.outputs[b]
                                ]
                        else:
                            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                                self.encoder_inputs, self.decoder_inputs, targets, self.target_weights, buckets,
                                lambda x,y: seq2seq_f(x,y,False),
                                softmax_loss_function = softmax_loss_function
                            )

                        params = tf.trainable_variables()
                        if not forward_only:
                            self.gradient_norms =[]
                            self.updates=[]
                            opt=tf.train.GradientDescentOptimizer(self.learning_rate)
                            for b in xrange(len(buckets)):
                                gradients = tf.gradients(self.losses[b], params)
                                clipped_gradient, norm = tf.clip_by_global_norm(gradients,max_gradient_norm)
                                self.gradient_norms.append(norm)
                                self.updates.append(opt.apply_gradients(zip(clipped_gradient,params), global_epoch =self.global_epoch))
                            self.saver =tf.train.Saver(tf.all_variables())
                    
                    def epoch (self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only):
                        encoder_size, decoder_size =self.buckets[bucket_id]
                        if len(encoder_inputs) != encoder_size:
                            raise ValueError("Encoder length must be equal to the one in buckets, " "%d != %d." % (len(encoder_inputs), encoder_inputs))
                        if len(decoder_inputs) != decoder_size:
                            raise ValueError("Decoder length must be equal to the one in bucket,"
                                           " %d != %d." % (len(decoder_inputs), decoder_size))
                        if len(target_weights) != decoder_size:
                            raise ValueError("Weights length must be equal to the one in bucket,"
                                            " %d != %d." % (len(target_weights), decoder_size))


                        input_feed = {}
                        for l in xrange(encoder_size):
                            input_feed[self.encoder_inpuuts[l].name] = encoder_inputs[l]
                        for l in xrange(decoder_size):
                            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
                            input_feed[self.target_weights[l].name] = target_weights[l]

                        last_target = self.decoder_inputs[decoder_size].name
                        input_feed[last_target] = np.zeros([self.batch_size], dtype = np.int32)

                        if not forward_only:
                            output_feed = [self.updates[bucket_id],
                                            self.gradient_norms[bucket_id],
                                            self.losses[bucket_id]]
                        else:
                            output_feed = [self.updates[bucket_id]]
                            for l in xrange(decoder_size):
                                output_feed.append(self.output[bucket_id][l])

                        outputs =session.run(output_feed, input_feed)
                        if not forward_only:
                            return outputs[1], outputs[2], None
                        else:
                            return None, outputs[0], outputs[1:]
                    
                    def get_batch(self, data, bucket_id):
                        encoder_size, decoder_size = self.buckets[bucket_id]
                        encoder_inputs, decoder_inputs = [],[]

                        for _ in xrange(self.batch_size):
                            encoder_inputs, decoder_input = random.choice(data[bucket_id])
                            encoder_pad = [data_utils,PAD_ID] * (encoder_size - len(encoder_input))
                            enocder_inputs.append(list(reversed(encoder_input + enocder_pad)))
                            decoder_pad_size = decoder_size = len(decoder_input) -1
                            decoder_inputs.append([data_utils.GO_ID] + decoder_input+ [data_utils.PAD_ID] * decoder_pad_size)
                        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [],[],[]
                        for length_idx in xrange(encoder_size):
                            batch_encoder_inputs.append(
                                np.array([encoder_inputs[batch_idx][length_idx]
                                        for batch_idx in xrange(self.batch_size)], dtype = np.int32))
                            batch_weights = np.ones(self.batch_size, dtype = np.float32)
                            for batch_idx in xrange(self.batch_size):
                                 if length_idx < decoder_size -1:
                                     target = decoder_inputs[batch_idx][length_idx + 1]
                                 if length_idx < decoder_size -1 or target == data_utils.PAD_ID:
                                     batch_weight[batch_idx]= 0.0
                            batch_weights.append(batch_weights)
                        return batch_encoder_inputs, batch_decoder_inputs, batch_weights



                        