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
import labels_diz as lbl
import numpy as np
import helper

#%%
#saver = tf.train.import_meta_graph('./exports/KITTI_segm/KITTI_segm-33.meta')
#saver.restore(sess,'./exports/KITTI_segm/KITTI_segm-33')

labels_diz = lbl.labels_diz
num_classes = len(labels_diz)
image_shape=(384,1216)

alfa = (127,) #semi-transparent
colors = np.array([label.color + alfa for label in labels_diz]).astype(np.uint8)

sess = tf.Session()

saver = tf.train.import_meta_graph('/media/avarfolomeev/storage/Data/Segmentation/net/my2-net-3029.meta')
saver.restore(sess,'/media/avarfolomeev/storage/Data/Segmentation/net/my2-net-3029')

model = tf.get_default_graph()

input_image = model.get_tensor_by_name('image_input:0')
keep_prob = model.get_tensor_by_name('keep_prob:0')
nn_output = model.get_tensor_by_name('layer3_up/BiasAdd:0')


logits = tf.reshape(nn_output,(-1,num_classes))

#%%

def segment_file(image_file):
    image0 = scipy.misc.imread(image_file)
    original_shape = image0.shape

    image = scipy.misc.imresize(image0, image_shape)

    image = cv2.GaussianBlur(image,(3,3),2)
    
    im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, input_image: [image]})

    res = im_softmax[0].reshape(out_shape)
    mx=np.argmax(res,2)

    out_colors = colors[mx]    

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
#data_folder='/media/D/DIZ/Datasets/KITTI/2011_10_03/2011_10_03_drive_0027_sync/'
#data_folder='/media/D/DIZ/Datasets/KITTI/2011_09_26/2011_09_26_drive_0084_sync/'
data_folder='/media/avarfolomeev/storage/Data/voxels/2018_03_08/2018_03_08_drive_0022_sync/'

road_name = 'Xroad'
overlay_name = 'Xoverlay'

try:
    os.makedirs(os.path.join(data_folder,road_name))
except:
    pass        

try:
    os.makedirs(os.path.join(data_folder,overlay_name))
except:
    pass        


out_shape = (image_shape[0], image_shape[1], num_classes)

l = glob(os.path.join(data_folder, 'image_02/data', '*.png'))

for im_file in l:
    im_out, mask = segment_file(im_file)
    out_file = im_file.replace('/data/','/../' + overlay_name + '/')
    scipy.misc.imsave(out_file, im_out)
    print(out_file)
    out_file = im_file.replace('/data/','/../' + road_name + '/')
    cv2.imwrite(out_file,mask)

#%%

