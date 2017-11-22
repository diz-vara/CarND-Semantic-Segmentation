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
from __future__ import print_function
import numpy as np

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
    
    input_image = model.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = model.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = model.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = model.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = model.get_tensor_by_name(vgg_layer7_out_tensor_name)

    
    return input_image, keep_prob,layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


#%%

    data_dir = './data'
    tf.reset_default_graph();
    img = tf.placeholder(tf.float32, name='image_input')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    l3 = tf.placeholder(tf.float32, name='layer3_out')
    l4 = tf.placeholder(tf.float32, name='layer4_out')
    l7 = tf.placeholder(tf.float32, name='layer7_out')
    
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        img, keep_prob, l3, l4, l7 = load_vgg(sess, vgg_path);
        #print(sess.run(img))
        
#%%

    import os.path
    import tensorflow as tf
    import helper
    import warnings
    from distutils.version import LooseVersion
    import project_tests as tests
    from __future__ import print_function
    import numpy as np

    tf.reset_default_graph();

    vgg_tag = 'vgg16'
    vgg_path='./data/vgg'
    a = np.random.randn(4,160,576,3)

    with tf.Session() as s1:
        tf.saved_model.loader.load(s1, [vgg_tag], vgg_path)
        model = s1.graph; #tf.get_default_graph(); #s1.graph
        
        keep_prob = model.get_tensor_by_name('keep_prob:0')
        layer_out = model.get_tensor_by_name('layer3_out:0')
        image_input = model.get_tensor_by_name('image_input:0')

        
        #metadata = tf.RunMetadata()
        run_options = tf.RunOptions(trace_level = tf.RunOptions.SOFTWARE_TRACE)
        o = s1.run(layer_out,feed_dict={image_input:a, keep_prob:1},
                  options = run_options)
                      #run_metadata = metadata)
            #writer.add_run_metadata(run_metadata, "step%d" % i);
            #writer.add_summary(summary, i)
        writer = tf.summary.FileWriter('/tmp/log/tf', s1.graph)
        writer.close()

#%%
    tf.reset_default_graph();
    vgg_path='./data/vgg'
    a = np.random.randn(4,160,576,3)

    with tf.Session() as s2:
        image_input, keep_prob,l3_o, l4_o, l7_o = load_vgg(s2, vgg_path);
        last = layers(l3_o, l4_o, l7_o, 2)
        #metadata = tf.RunMetadata()
        s2.run(tf.global_variables_initializer())
        run_options = tf.RunOptions(trace_level = tf.RunOptions.SOFTWARE_TRACE)
        o = s2.run(last,feed_dict={image_input:a, keep_prob:1})
                  #options = run_options)
                      #run_metadata = metadata)
            #writer.add_run_metadata(run_metadata, "step%d" % i);
            #writer.add_summary(summary, i)
        writer = tf.summary.FileWriter('/tmp/log/tf', s2.graph)
        writer.close()
        
