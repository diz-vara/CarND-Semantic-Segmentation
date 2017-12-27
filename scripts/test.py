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
import labels


#%%
#saver = tf.train.import_meta_graph('./exports/KITTI_segm/KITTI_segm-33.meta')
#saver.restore(sess,'./exports/KITTI_segm/KITTI_segm-33')

sess = tf.Session()
labels = labels.labels
nclasses = len(labels)



saver = tf.train.import_meta_graph('/media/D/DIZ/CityScapes/net/my-net-12.meta')
saver.restore(sess,'/media/D/DIZ/CityScapes/net/my-net-12')

graph=tf.get_default_graph()
keep_prob = graph.get_tensor_by_name('keep_prob:0')
image_in = graph.get_tensor_by_name('image_input:0')
image_shape = (160, 512)
nn_out = graph.get_tensor_by_name('layer3_up/BiasAdd:0')
logits = tf.reshape(nn_out,(-1,nclasses))

#%%
data_folder='/media/D/DIZ/Datasets/KITTI/2011_09_26/2011_09_26_drive_0117_sync/'
#data_folder='/media/D/DIZ/Datasets/KITTI/2011_10_03/2011_10_03_drive_0042_sync/'
l = glob(os.path.join(data_folder, 'image_02/data', '*.png'))
#%%

def segment_file(image_file):
    image0 = scipy.misc.imread(image_file)
    original_shape = image0.shape

    image = scipy.misc.imresize(image0, image_shape)
    im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, image_in: [image]})

    res = im_softmax[0].reshape(out_shape)
    mx=np.argmax(res,2)

    alfa = (127,)
    out_colors = np.zeros((image_shape[0],image_shape[1],4),dtype=np.uint8)
    for y in range(image_shape[0]):
        for x in range(image_shape[1]):
            out_colors[y,x,:] = labels[mx[y,x]].color + alfa

    #out_image = scipy.misc.toimage(mx, mode = 'L')
    out_image = cv2.resize(mx, (original_shape[1], original_shape[0]), 
                                interpolation=cv2.INTER_NEAREST)

    out_colors = scipy.misc.imresize(out_colors,original_shape)
    colors_img = scipy.misc.toimage(out_colors, mode="RGBA")

    street_im = scipy.misc.toimage(image0)
    street_im.paste(colors_img,box=None,mask=colors_img)


    
    return street_im, out_image

#plt.imshow(street_im)
#%%

try:
    os.makedirs(os.path.join(data_folder,'Croad'))
    os.makedirs(os.path.join(data_folder,'Coverlay'))
except:
    pass        

out_shape = (image_shape[0], image_shape[1], 35)

l = glob(os.path.join(data_folder, 'image_02/data', '*.png'))

for im_file in l:
    im_out, mask = segment_file(im_file)
    out_file = im_file.replace('/data/','/../Coverlay/')
    scipy.misc.imsave(out_file, im_out)
    print(out_file)
    out_file = im_file.replace('/data/','/../Croad/')
    cv2.imwrite(out_file,mask)
