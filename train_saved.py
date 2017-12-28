# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:12:51 2017

@author: avarfolomeev
"""

import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import labels as lbl
import numpy as np
import helper


#%%
def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
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
    
    #sess.run(tf.global_variables_initializer())
    #saver = tf.train.Saver();

    #lr = sess.run(learning_rate)
    #merged = tf.summary.merge_all()
    lr = 1e-4
    min_loss = 1e9
    for epoch in range (epochs):
        print ('epoch {}  '.format(epoch))
        print(" LR = {:f}".format(lr))     
        for image, label in get_batches_fn(batch_size):
            summary, loss = sess.run([train_op, cross_entropy_loss],
                                     feed_dict={input_image:image, 
                                                correct_label:label,
                                     keep_prob:0.5, learning_rate:lr})
        #writer.add_summary(summary, epoch)                          
        lr = lr * 0.9                            
        print(" Loss = {:g}".format(loss))     
        print()                        
        if (loss < min_loss):
            print("saving at step {:d}".format(epoch))     
            min_loss = loss;
            saver.save(sess, '/media/D/DIZ/CityScapes/net/my2-net',global_step=epoch)
            
#%%
#%%

#def retrain():
    
tf.reset_default_graph()

data_dir = './data'
runs_dir = './runs'
timestamp = time.strftime("%Y%m%d_%H%M%S");

export_dir = './exports/' + timestamp;

labels = lbl.labels
num_classes = len(labels)
image_shape=(256,512)

epochs = 50
batch_size = 8


alfa = (127,) #semi-transparent
colors = np.array([label.color + alfa for label in labels]).astype(np.uint8)

config = tf.ConfigProto(
   gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8),
   device_count = {'GPU': 1}
)
sess = tf.Session(config = config)
#saver = tf.train.Saver()
saver = tf.train.import_meta_graph('/media/D/DIZ/CityScapes/net/my-net-41.meta')
saver.restore(sess,'/media/D/DIZ/CityScapes/net/my-net-41')

graph=tf.get_default_graph()
keep_prob = graph.get_tensor_by_name('keep_prob:0')
image_in = graph.get_tensor_by_name('image_input:0')
nn_output = graph.get_tensor_by_name('layer3_up/BiasAdd:0')

assert(nn_output.shape[-1] == num_classes)


logits = tf.reshape(nn_output,(-1,num_classes))

    
correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes],
                               name = 'correct_label')
learning_rate = tf.placeholder(tf.float32, name='learning_rate')                                       

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=correct_label,
                                                        logits = logits,
                                                        name = "cross-ent")
loss = tf.reduce_mean(cross_entropy);


get_batches_fn = helper.gen_batch_function('/media/D/DIZ/CityScapes',
                                           image_shape, num_classes)

train_op=graph.get_collection('train_op')


print('training')
train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
         loss, image_in, correct_label, keep_prob, learning_rate)                                          




#if __name__ == '__main__':
#    retrain()
            