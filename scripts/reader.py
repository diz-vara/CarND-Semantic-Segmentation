# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:49:00 2017

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



export_dir = './exports/20171121_100208'
vgg_path = './data/vgg'
vgg_tag = 'vgg16'
serv_tag = 'serve'

config = tf.ConfigProto(
   gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5),
   device_count = {'GPU': 1}
)

sess = tf.Session(config = config)

tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
out = sess.graph.get_tensor_by_name("layer_3_up:0")
tf.Print(out)                           
    #writer = tf.summary.FileWriter('/tmp/log/tf', sess.graph)
    #writer.close()
    
    
