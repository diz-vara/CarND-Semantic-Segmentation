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
import labels as lbl
import numpy as np
import helper

#%%
#saver = tf.train.import_meta_graph('./exports/KITTI_segm/KITTI_segm-33.meta')
#saver.restore(sess,'./exports/KITTI_segm/KITTI_segm-33')

labels = lbl.labels
num_classes = len(labels)
image_shape=(160,512)

alfa = (127,) #semi-transparent
colors = np.array([label.color + alfa for label in labels]).astype(np.uint8)

sess = tf.Session()

saver = tf.train.import_meta_graph('/media/D/DIZ/CityScapes/net/my2-net-18.meta')
saver.restore(sess,'/media/D/DIZ/CityScapes/net/my2-net-18')

graph=tf.get_default_graph()
keep_prob = graph.get_tensor_by_name('keep_prob:0')
image_in = graph.get_tensor_by_name('image_input:0')
nn_output = graph.get_tensor_by_name('layer3_up/BiasAdd:0')

assert(nn_output.shape[-1] == num_classes)


logits = tf.reshape(nn_output,(-1,num_classes))

#%%

def segment_file(image_file):
    image0 = scipy.misc.imread(image_file)
    original_shape = image0.shape

    image = scipy.misc.imresize(image0, image_shape)
    im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, image_in: [image]})

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
data_folder='/media/D/DIZ/Datasets/KITTI/2011_10_03/2011_10_03_drive_0027_sync/'

road_name = 'Croad'
overlay_name = 'Coverlay'

try:
    os.makedirs(os.path.join(data_folder,road_name))
except:
    pass        

try:
    os.makedirs(os.path.join(data_folder,overlay_name))
except:
    pass        


out_shape = (image_shape[0], image_shape[1], 35)

l = glob(os.path.join(data_folder, 'image_02/data', '*.png'))

for im_file in l:
    im_out, mask = segment_file(im_file)
    out_file = im_file.replace('/data/','/../' + overlay_name + '/')
    scipy.misc.imsave(out_file, im_out)
    print(out_file)
    out_file = im_file.replace('/data/','/../' + road_name + '/')
    cv2.imwrite(out_file,mask)

#%%
out_graph = '/media/D/DIZ/CityScapes/net/my2-net-18-frozen.pb'
out_name = 'layer3_up/BiasAdd'

graph_def = tf.get_default_graph().as_graph_def()

output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            graph_def, # The graph_def is used to retrieve the nodes 
            [out_name] # The output node names are used to select the usefull nodes
        )


with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

print("%d ops in the input graph." % len(graph_def.node))
print("%d ops in the final graph." % len(output_graph_def.node))


