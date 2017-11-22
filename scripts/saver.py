# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:36:22 2017

@author: avarfolomeev
"""

import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import tensorflow.contrib.slim as slim


def save():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'


    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:


        # TODO: Save inference data using helper.save_inference_samples
        print('SAVING')
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, 
                                      logits, keep_prob, image_in)

