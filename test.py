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

alfa = (127,) #semi-transparent
colors = np.array([label.color + alfa for label in labels_diz]).astype(np.uint8)

sess = tf.Session()

load_net = '/media/avarfolomeev/storage/Data/Segmentation/net/my2-net-25794'

saver = tf.train.import_meta_graph(load_net + '.meta')
saver.restore(sess,load_net)


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
dataset = 'LA'

if dataset == 'London':
    data_folder='/media/avarfolomeev/storage/Data/voxels/2018_03_08/L21/'
    image_shape=(384,640)
    dataname = 'image_02/data/'
    l = glob(os.path.join(data_folder, dataname, '*.png'))
elif dataset == 'CN':
    data_folder='/media/avarfolomeev/storage/Data/CN/2017_11_08_06_32_41/'
    image_shape=(352,640)
    dataname = 'Collected/'
    l = glob(os.path.join(data_folder, dataname, '*.jpg'))
elif dataset == 'KITTI':
    #data_folder='/media/D/DIZ/Datasets/KITTI/2011_10_03/2011_10_03_drive_0027_sync/'
    data_folder='/media/D/DIZ/Datasets/KITTI/2011_09_26/2011_09_26_drive_0084_sync/'
    image_shape=(192,608)
    dataname = 'image_02/data/'
    l = glob(os.path.join(data_folder, dataname, '*.png'))
elif dataset == 'CS':
    #data_folder='/media/D/DIZ/Datasets/KITTI/2011_10_03/2011_10_03_drive_0027_sync/'
    data_folder='/media/D/DIZ/Datasets/KITTI/2011_09_26/2011_09_26_drive_0084_sync/'
    image_shape=(320,640)
    dataname = 'image_02/data/'
    l = glob(os.path.join(data_folder, dataname, '*.png'))
elif dataset == 'spb-R1':
    data_folder='/media/avarfolomeev/storage/Data/voxels/20180525/ride01/'
    image_shape=(384,640)
    dataname = 'image_02/data/'
    l = glob(os.path.join(data_folder, dataname, '*.png'))
elif dataset == 'NavInfo':
    data_folder='/media/avarfolomeev/storage/Data/voxels/NavInfo/'
    image_shape=(384,640)
    dataname = 'Screens/'
    l = glob(os.path.join(data_folder, dataname, '*.png'))
elif dataset == 'LA':
    data_folder='/media/avarfolomeev/storage/Data/voxels/20180908/test8_6/argus_cam_5/'
    image_shape=(384,640)
    dataname = 'data/'
    l = glob(os.path.join(data_folder, dataname, '*.png'))
    



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


for im_file in l:
    im_out, mask = segment_file(im_file)
    out_file = im_file.replace(dataname,overlay_name + '/')
    scipy.misc.imsave(out_file, im_out)
    print(out_file)
    out_file = im_file.replace(dataname,road_name + '/').replace('.jpg','.png')
    cv2.imwrite(out_file,mask)

#%%

