from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
from six.moves import urllib

from tensorflow.python.platform import gfile

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD,_GO,_EOS,_UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID =2
UNK_ID=3

_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


def basic_tokenizer(sentence):
    words =[]

    for space_sperated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_sperated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size, tokenizer=None, normalize_digits=True):
    if not gfile.Exists(vocabulary_path):
        counter =0
        for line in f:
            counter += 1
            if counter % 100000 == 0:
                print(" processing line %d" %counter)
            tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
            for w in tokens:
                word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else word
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse =True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_path, mode ="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w+b"\n")

def intialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab=[]
        with gFile.GFile(vcabulary_path, mode ="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x,y) for (y,x) in enumerate(rev_voacb)])
        vocab = dict([(x,y) for (y,x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocubulary file %s not found,", vocabulary_path)

def sentence_to_token_ids(sentence, vocabulary, tokenizer =None, normalize_digits =True):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]

def data_to_token_ids(data_path, target_path,vocabulary_path, tokenizer=None, normalize_digits=True):
        if not ffile.Exists(target_path):
            vocab, _ = intialize_vocabulary(vocabulary_path)
            with gfile.GFile(data_path, mode="rb") as data_file:
                with gfile.GFile(target_path, mode="w") as tokens_file:
                    counter =0
                    for line in data_file:
                        counter +=1
                        token_ids = sentence_token_ids(line, vocab, tokenizer, normalize_digits)
                        tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n" )

def prepare_data(working_directory, train_enc, test_enc, test_dec, enc_vocabulary_size, dec_vocabulary_size, tokenizer=None):
        enc_vocab_path = os.path.join(working_directory, "vocab%d.enc" % enc_vocabulary_size)
        dec_vocab_path = os.path.join(working_directory, "vocab%d.dec" % dec_vocabulary_size)
        create_vocabulary(enc_vocab_path, train_enc, enc_vocabulary_size, tokenizer)

        enc_train_ids_path =test_enc + (".ids%d" % enc_vocabulary_size)
        dec_train_ids_path =train_dec + (".ids%d" % dec_vocabulary_size)
        data_to_token_ids(test_enc, enc_dev_ids_path, enc_vocab_path, tokenizer)
        data_to_token_ids(test_dec, dec_train_ids_path, dec_vocab_path,tokenizer)

        enc_dev_ids_path =test_enc + (".ids%d" % enc_vocab_path)
        dec_dev_ids_path =test_dec + (".ids%d" % dec_vocabulary_size)
        data_to_token_ids(test_enc,enc_dev_ids_path, enc_vocab_path,tokenizer)
        data_to_token_ids(test_dec,dec_dev_ids_path, dec_vocab_path,tokenizer)
        return (enc_train_ids_path, dec_train_ids_path, enc_dev_ids_path,dec_dev_ids_path,enc_vocab_path,enc_vocab_path,dec_vocab_path)
                