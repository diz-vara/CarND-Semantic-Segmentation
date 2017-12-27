# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 09:29:12 2017

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


#%%
#saver = tf.train.import_meta_graph('./exports/KITTI_segm/KITTI_segm-33.meta')
#saver.restore(sess,'./exports/KITTI_segm/KITTI_segm-33')

sess = tf.Session()

saver = tf.train.import_meta_graph('/media/D/DIZ/CityScapes/net/net-0.meta')
saver.restore(sess,'/media/D/DIZ/CityScapes/net/net-0')

graph=tf.get_default_graph()
keep_prob = graph.get_tensor_by_name('keep_prob:0')
image_in = graph.get_tensor_by_name('image_input:0')
image_shape = (256, 512)
nn_out = graph.get_tensor_by_name('layer3_up/BiasAdd:0')
logits = tf.reshape(nn_out,(-1,3))

#%%
data_folder='/media/D/DIZ/Datasets/KITTI/2011_09_26/2011_09_26_drive_0117_sync/'
#data_folder='/media/D/DIZ/Datasets/KITTI/2011_10_03/2011_10_03_drive_0042_sync/'
l = glob(os.path.join(data_folder, 'image_02/data', '*.png'))
#%%
#image_file = l[490]
def segment_file(image_file):
    image = scipy.misc.imread(image_file)
    original_shape = image.shape

    image = scipy.misc.imresize(image, image_shape)
    im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, image_in: [image]})
    road_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    road_segmentation = (road_softmax > 0.6).reshape(image_shape[0], image_shape[1], 1)
    road_mask = np.dot(road_segmentation, np.array([[0, 255, 0, 127]],dtype=np.uint8))
    road_mask = scipy.misc.toimage(road_mask, mode="RGBA")
    out_image = np.dot(road_segmentation, np.array([[255, 0, 255]],dtype=np.uint8))  
    not_road = out_image[:,:,2] == 0
    out_image[not_road,2] = 255
    out_image = scipy.misc.imresize(out_image, original_shape)
    
    other_softmax = im_softmax[0][:, 2].reshape(image_shape[0], image_shape[1])
    other_segmentation = (other_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    other_mask = np.dot(other_segmentation, np.array([[200, 0, 0, 127]],dtype=np.uint8))
    other_mask = scipy.misc.toimage(other_mask, mode="RGBA")
    
    street_im = scipy.misc.toimage(image)
    street_im.paste(road_mask, box=None, mask=road_mask)
    street_im.paste(other_mask, box=None, mask=other_mask)
    return street_im, out_image

#plt.imshow(street_im)
#%%

try:
    os.makedirs(os.path.join(data_folder,'road'))
    os.makedirs(os.path.join(data_folder,'overlay'))
except:
    pass        

l = glob(os.path.join(data_folder, 'image_02/data', '*.png'))

for im_file in l:
    im_out, mask = segment_file(im_file)
    out_file = im_file.replace('/data/','/../overlay/')
    scipy.misc.imsave(out_file, im_out)
    print(out_file)
    out_file = im_file.replace('/data/','/../road/')
    cv2.imwrite(out_file,mask)
