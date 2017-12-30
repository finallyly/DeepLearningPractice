#-*-coding:utf8-*-
#########################################################################
#   Copyright (C) 2017 All rights reserved.
# 
#   FileName:Seq2Seq.py
#   Creator: yuliu1finally@gmail.com
#   Time:12/21/2017
#   Description:
#
#   Updates:
#
#########################################################################
#!/usr/bin/python
# please add your code here!
import numpy as np;
import time;
import helper;
source_path="data/letters_source.txt";
target_path="data/letters_target.txt";
source_sentences = helper.load_data(source_path);
target_sentences = helper.load_data(target_path);

def extract_character_vocab(data):
    special_words = ['<PAD>','<UNK>','<GO>','<EOS>'];
    set_words = set([character for line in data.split('\n') for character in line]);
    set_words = set([character for line in data.split('\n') for character in line])
    int_to_vocab = {word_i: word for word_i, word in enumerate(special_words + list(set_words))}
    vocab_to_int = {word: word_i for word_i, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int;
# Build int2letter and letter2int dicts
source_int_to_letter, source_letter_to_int = extract_character_vocab(source_sentences)
target_int_to_letter, target_letter_to_int = extract_character_vocab(target_sentences)
source_letter_ids = [[source_letter_to_int.get(letter, source_letter_to_int['<UNK>']) for letter in line] for line in source_sentences.split('\n')]
target_letter_ids = [[target_letter_to_int.get(letter, target_letter_to_int['<UNK>']) for letter in line] + [target_letter_to_int['<EOS>']] for line in target_sentences.split('\n')]

print("Example source sequence")
print(source_letter_ids[:3])
print("\n")
print("Example target sequence")
print(target_letter_ids[:3])

from distutils.version import LooseVersion
import tensorflow as tf
from tensorflow.python.layers.core import Dense


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.1'), 'Please use TensorFlow version 1.1 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Number of Epochs
epochs = 60
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 50
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 15
decoding_embedding_size = 15
# Learning Rate
learning_rate = 0.001

def get_model_inputs():
    input_data = tf.placeholder(tf.int32,[None,None],name="input");
    targets = tf.placeholder(tf.int32,[None,None],name="targets");
    lr = tf.placeholder(tf.float32,name="learning_rate");
    target_sequence_length = tf.placeholder(tf.int32,(None,),name="target_sequence_length");
    max_target_sequence_length = tf.reduce_max(target_sequence_length,name='max_target_len');
    source_sequence_length = tf.placeholder(tf.in32,(None,),name="source_sequence_length");
    return input_data,targets,lr,target_sequence_length,max_target_sequence_length,source_sequence_length;


def encoding_layer(input_data,rnn_size,num_layers,
                   source_sequence_length,source_vocab_size,
                   encoding_embedding_size):
    #Encoder embedding
    enc_embed_input = tf.contrib.layers.embed_sequence(input_data,source_vocab_size,encoding_embedding_size);

    #RNN cell
    def make_cell(rnn_size):
        enc_cell = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_normal_initializer(-0.1,0.1,seed=2));
        return  enc_cell;

    enc_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)]);
    enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_embed_input, sequence_length=source_sequence_length,
                                              dtype=tf.float32);
    return enc_output, enc_state;


def decoding_layer(target_letter_to_int, decoding_embedding_size,num_layers,rnn_size,
                   target_sequence_length,max_target_sequence_length,enc_state,dec_input):
    #1. Decoder embedding
    target_vocab_size = len(target_letter_to_int);
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size,decoding_embedding_size]));
    dec_embed_input =tf.nn.embedding_lookup(dec_embeddings,dec_input);

    #2. Construct the decoder cell

    def make_cell(rnn_size):
        dec_cell = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_normal_initializer(-0.1,0.1,seed=2));
        return dec_cell;

    dec_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)]);

    #3 Dense layer to translate the decoder's output at each time
    # step into a choice from the target vocabulary

    output_layer = Dense(target_vocab_size,kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1));

    #4. Set up a training decoder and inference decoder
    # Training Decoder
    with tf.variable_scope("decode"):
        #Helper for the training process. Used by BasicDecoder to read inputs
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,sequence_length=target_sequence_length
                                            ,time_major=False);
        #Basic decoder
       training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,training_helper,enc_state,output_layer);

        #Perform dynamic decoding using the decoder
        training_decoder_output = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                    impute_finished=True,
                                                                    maximum_iterations=max_target_sequence_length)[0];



    #5 Inference Decoder
    # Reuses the same parameters trained by training process

    with tf.variable_scope("decode",reuse=True):
        start_tokens = tf.tile(tf.constant([target_letter_to_int['<GO>']],dtype=tf.int32),
                               [batch_size],name='start_tokens');

        #Helper for the inference process

        inference_helper= tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, start_tokens,target_letter_to_int['<EOS>']);

        inference_decoder=tf.contrib.seq2seq.BasicDecoder(dec_cell,inference_helper,enc_state,output_layer);

        # Perform dynamic decoding using the decoder
        inference_decoder_output = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                     impute_finished=True,
                                                                     maximum_iterations=max_target_sequence_length)[0];

    return training_decoder_output, inference_decoder_output






