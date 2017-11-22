# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 20:00:50 2017

@author: diz
"""
import tensorflow as tf

tf.reset_default_graph();

train_op = tf.constant(0)
cross_entropy_loss = tf.constant(10.11)
input_image = tf.placeholder(tf.float32, name='input_image')
correct_label = tf.placeholder(tf.float32, name='correct_label')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
learning_rate = tf.placeholder(tf.float32, name='learning_rate')



def train_nn(sess, keep_prob):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.global_variables_initializer())
    #lr = sess.run(learning_rate)
    
    #k_prob = sess.run(keep_prob)
    #l_rate = sess.run(learning_rate)
    sess.run(train_op,feed_dict={keep_prob:keep_prob})
        
    

with tf.Session() as sess:
    train_nn(sess, sess.run(keep_prob))
            