#-*-coding:utf8-*-
#########################################################################
#   Copyright (C) 2017 All rights reserved.
# 
#   FileName:SemiSupervisedGAN.py
#   Creator: yuliu1finally@gmail.com
#   Time:12/19/2017
#   Description:
#
#   Updates:
#
#########################################################################
#!/usr/bin/python
# please add your code here!
import pickle as pkl;
import time;
import matplotlib.pyplot as plt;
import numpy as np;
from scipy.io import loadmat;
import tensorflow as tf;

data_dir = 'data/';

trainset=loadmat(data_dir+'train_32x32.mat');
testset=loadmat(data_dir+'test_32x32.mat');


def scale(x, feature_range=(-1, 1)):
    # scale to (0, 1)
    x = ((x - x.min()) / (255 - x.min()))

    # scale to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x


class Dataset:
    def __init__(self, train, test, val_frac=0.5, shuffle=True, scale_func=None):
        split_idx = int(len(test['y']) * (1 - val_frac))
        self.test_x, self.valid_x = test['X'][:, :, :, :split_idx], test['X'][:, :, :, split_idx:]
        self.test_y, self.valid_y = test['y'][:split_idx], test['y'][split_idx:]
        self.train_x, self.train_y = train['X'], train['y']
        # The SVHN dataset comes with lots of labels, but for the purpose of this exercise,
        # we will pretend that there are only 1000.
        # We use this mask to say which labels we will allow ourselves to use.
        self.label_mask = np.zeros_like(self.train_y)
        self.label_mask[0:1000] = 1

        self.train_x = np.rollaxis(self.train_x, 3)
        self.valid_x = np.rollaxis(self.valid_x, 3)
        self.test_x = np.rollaxis(self.test_x, 3)

        if scale_func is None:
            self.scaler = scale
        else:
            self.scaler = scale_func
        self.train_x = self.scaler(self.train_x)
        self.valid_x = self.scaler(self.valid_x)
        self.test_x = self.scaler(self.test_x)
        self.shuffle = shuffle

    def batches(self, batch_size, which_set="train"):
        x_name = which_set + "_x"
        y_name = which_set + "_y"

        num_examples = len(getattr(dataset, y_name))
        if self.shuffle:
            idx = np.arange(num_examples)
            np.random.shuffle(idx)
            setattr(dataset, x_name, getattr(dataset, x_name)[idx])
            setattr(dataset, y_name, getattr(dataset, y_name)[idx])
            if which_set == "train":
                dataset.label_mask = dataset.label_mask[idx]

        dataset_x = getattr(dataset, x_name)
        dataset_y = getattr(dataset, y_name)
        for ii in range(0, num_examples, batch_size):
            x = dataset_x[ii:ii + batch_size]
            y = dataset_y[ii:ii + batch_size]

            if which_set == "train":
                # When we use the data for training, we need to include
                # the label mask, so we can pretend we don't have access
                # to some of the labels, as an exercise of our semi-supervised
                # learning ability
                yield x, y, self.label_mask[ii:ii + batch_size]
            else:
                yield x, y


def model_inputs(real_dim,z_dim):
    inputs_real = tf.placeholder(tf.float32,(None,real_dim[0],real_dim[1],real_dim[2]),name='input_real');
    inputs_z = tf.placeholder(tf.float32,(None,z_dim),name='input_z');
    y=tf.placeholder(tf.int32,(None),name='y');
    label_mask = tf.placeholder(tf.int32,(None),name='label_mask');
    return inputs_real,inputs_z,y,label_mask;

def generator(z,output_dim,reuse=False,alpha=0.2,training=True,size_mult=128):
    with tf.variable_scope("generator",reuse=reuse):
        x1=tf.layers.dense(z,4*4*size_mult*4);
        x1=tf.reshape(x1,(-1,4,4,size_mult*4));
        x1=tf.layers.batch_normalization(x1,training=training);
        x1=tf.maximum(alpha*x1,x1);
        x2=tf.layers.conv2d_transpose(x1,size_mult*2,5,strides=2,padding="same");
        x2 = tf.layers.batch_normalization(x2, training=training)
        x2 = tf.maximum(alpha * x2, x2);
        x3 = tf.layers.conv2d_transpose(x2, size_mult, 5, strides=2, padding='same')
        x3 = tf.layers.batch_normalization(x3, training=training)
        x3 = tf.maximum(alpha * x3, x3);
        # Output layer
        logits = tf.layers.conv2d_transpose(x3, output_dim, 5, strides=2, padding='same')
        out = tf.tanh(logits)
        return out;


def discriminator(x,reuse=False,alpha=0.2,drop_rate=0.0,num_classes=10,size_mult=64):
    with tf.variable_scope('discriminator',reuse=reuse):
        x=tf.layers.dropout(x,rate=drop_rate/2.5);

        x1=tf.layers.conv2d(x,size_mult,3,strides=2,padding='same');
        relu1 = tf.maximum(alpha * x1, x1)
        relu1 = tf.layers.dropout(relu1, rate=drop_rate)


    x2 = tf.layers.conv2d(relu1, size_mult, 3, strides=2, padding='same')
    bn2 = tf.layers.batch_normalization(x2, training=True)
    relu2 = tf.maximum(alpha * x2, x2)

    x3 = tf.layers.conv2d(relu2, size_mult, 3, strides=2, padding='same')
    bn3 = tf.layers.batch_normalization(x3, training=True)
    relu3 = tf.maximum(alpha * bn3, bn3)
    relu3 = tf.layers.dropout(relu3, rate=drop_rate)

    x4 = tf.layers.conv2d(relu3, 2 * size_mult, 3, strides=1, padding='same')
    bn4 = tf.layers.batch_normalization(x4, training=True)
    relu4 = tf.maximum(alpha * bn4, bn4)

    x5 = tf.layers.conv2d(relu4, 2 * size_mult, 3, strides=1, padding='same')
    bn5 = tf.layers.batch_normalization(x5, training=True)
    relu5 = tf.maximum(alpha * bn5, bn5)

    x6 = tf.layers.conv2d(relu5, 2 * size_mult, 3, strides=2, padding='same')
    bn6 = tf.layers.batch_normalization(x6, training=True)
    relu6 = tf.maximum(alpha * bn6, bn6)
    relu6 = tf.layers.dropout(relu6, rate=drop_rate)

    x7 = tf.layers.conv2d(relu5, 2 * size_mult, 3, strides=1, padding='valid')
    # Flatten it by global average pooling
    features = tf.reduce_mean(relu7, (1, 2))
    # Set class_logits to be the inputs to a softmax distribution over the different classes
    class_logits = tf.layers.dense(features, num_classes + extra_class)



