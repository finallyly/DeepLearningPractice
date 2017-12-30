#-*-coding:utf8-*-
#########################################################################
#   Copyright (C) 2017 All rights reserved.
# 
#   FileName:MINIST_GAN.py
#   Creator: yuliu1finally@gmail.com
#   Time:12/17/2017
#   Description:
#
#   Updates:
#
#########################################################################
#!/usr/bin/python
# please add your code here!

import pickle as pkl;
import numpy as np;
import tensorflow as tf;
import matplotlib.pyplot as plt;
from tensorflow.examples.tutorials.mnist import input_data;
mnist = input_data.read_data_sets('MNIST_data');

def model_inputs(real_dim,z_dim):
    inputs_real = tf.placeholder(tf.float32,[None,real_dim],name='input_real');
    inputs_z = tf.placeholder(tf.float32,(None,z_dim),name='input_z');
    return inputs_real,inputs_z;

def generator(z,out_dim,n_units=128,reuse=False,alpha=0.01):
    with tf.variable_scope('generator',reuse=reuse):
        #Hidden layer
        h1=tf.layers.dense(z,n_units,activation=None);
        #Leakly ReLU
        h1 = tf.maximum(alpha*h1,h1);

        #Logits and tanh output
        logits = tf.layers.dense(h1,out_dim, activation=None);
        out = tf.tanh(logits);
        return out;

def discriminator(x,n_units=128,reuse=False,alpha=0.01):
    with tf.variable_scope("discriminator",reuse=reuse):
        # Hidden layer
        h1 = tf.layers.dense(x,n_units,activation=None);
        #Leaky Relu
        h1 = tf.maximum(alpha*h1,h1);

        logits = tf.layers.dense(h1,1,activation=None);
        out = tf.sigmoid(logits);
        return  out,logits;

# Size of input image to discriminator
input_size = 784;
#Size of latent vector to generator
z_size = 100;
# size of hidden layers in generator and discriminator
g_hidden_size = 128;
d_hidden_size = 128;
#Leaky factor for leaky ReLu
alpha=0.01;
#Smoothing
smooth = 0.1;

tf.reset_default_graph();
#Create our input placeholders
input_real,input_z = model_inputs(input_size,z_size);

#Build the model
g_model = generator(input_z,input_size,n_units=g_hidden_size,alpha=alpha);
d_model_real, d_logits_real = discriminator(input_real,n_units=d_hidden_size,alpha=alpha);
g_model_fake, d_logits_fake = discriminator(g_model,n_units=d_hidden_size,reuse=True,alpha=alpha);

#Calculate losses
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                     labels=tf.ones_like(d_logits_real)*(1-smooth)));

d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                     labels=tf.zeros_like(d_logits_fake)));
d_loss = d_loss_real+d_loss_fake;

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.ones_like(d_logits_fake)));

#Optimizers
learning_rate = 0.002;
# Get the trainable_variables
t_vars = tf.trainable_variables();
g_vars = [var for var in t_vars if var.name.startswith("generator")];
d_vars = [var for var in t_vars if var.name.startswith("discriminator")];
d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss,var_list=d_vars);
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss,var_list=g_vars);


batch_size = 100;
epochs = 100;
samples = [];
losses = [];

saver = tf.train.Saver(var_list=g_vars);
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer());
    for e in range(epochs):
        for ii in range(mnist.train.num_examples/batch_size):
            batch = mnist.train.next_batch(batch_size);

            # Get images,reshape and rescale to pass to D
            batch_images = batch[0].reshape((batch_size,784));
            batch_images = batch_images*2-1;

            # Sampel random noise for G
            batch_z = np.random.uniform(-1,1,size=(batch_size,z_size));

            #Run optimizers
            _ = sess.run(d_train_opt,feed_dict={input_real:batch_images,input_z:batch_z});
            _ =sess.run(g_train_opt,feed_dict={input_z:batch_z});

        # at the end of each epoch, get the losses and print them out
        train_loss_d = sess.run(d_loss,{input_real:batch_images,input_z:batch_z});
        train_loss_g = g_loss.eval({input_z: batch_z});

        print("Epoch {}/{}".format(e+1,epochs),
              "Discriminator Loss:{:.4f}...".format(train_loss_d),
              "Generator Loss:{:.4f}".format(train_loss_g));

        # Save losses to view after training
        losses.append((train_loss_d,train_loss_g));

        #Sample from generator  as we an training for viewing afterwards
        sample_z = np.random.uniform(-1, 1, size=(16, z_size));
        gen_samples = sess.run(generator(input_z,input_size,n_units=g_hidden_size,reuse=True,alpha=alpha),
                               feed_dict={input_z:sample_z});

        samples.append(gen_samples);
        saver.save(sess,"./checkpoints/generator.ckpt");

# Save training generator samples
with open('train_samples.pkl','wb') as f:
    pkl.dump(samples,f);

fig,ax = plt.subplots();
losses = np.array(losses);
plt.plot(losses.T[0],label="Discriminator");
plt.plot(losses.T[1],label="Generator");
plt.title("Training Losses");
plt.legend();
'''
rows, cols = 10, 6
fig, axes = plt.subplots(figsize=(7,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

for sample, ax_row in zip(samples[::int(len(samples)/rows)], axes):
    for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
        ax.imshow(img.reshape((28,28)), cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
plt.show();
def view_samples(epoch,samples):
    fig,axes = plt.subplots(figsize=(7,7),nrows=4,ncols=4,sharey=True,sharex=True);
    for ax,img in zip(axes.flatten(),samples[epoch]):
        ax.xaxis.set_visible(False);
        ax.yaxis.set_visible(False);
        im=ax.imshow(img.reshape(28,28),cmap="Greys_r");
    plt.show();
    return fig,axes;
with open('train_samples.pkl','rb') as f:
    samples=pkl.load(f);
_ = view_samples(-1,samples);

saver = tf.train.Saver(var_list=g_vars);
with  tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint("checkpoints"));
    sample_z = np.random.uniform(-1,1,size=(16,z_size));
    gen_samples=sess.run(
     generator(input_z,input_size,n_units=g_hidden_size,reuse=True,alpha=alpha),
    feed_dict={input_z:sample_z});

_=view_samples(0,[gen_samples])
'''






