# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 12:18:29 2017

@author: diz
"""
#%%
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


#%%
def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    model = tf.get_default_graph()
    print("model",model)
    
    input_image = model.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = model.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = model.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = model.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = model.get_tensor_by_name(vgg_layer7_out_tensor_name)

    
    return input_image, keep_prob,layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


#%%
    data_dir = './data'
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        vgg_input_tensor_name = 'image_input:0'

        tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
        model = tf.get_default_graph()
        input_image = model.get_tensor_by_name(vgg_input_tensor_name)
        #print(model)
