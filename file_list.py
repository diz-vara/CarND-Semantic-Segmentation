# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 14:33:20 2017

@author: avarfolomeev
"""
import os

def get_image_and_labels_list(root_path, label_path, image_path=''):
    image_list = []
    labels_list = []
    if (len(image_path) < 1):
        image_path = root_path
        
    for root, subdirs, files in os.walk(image_path):
    
        for file in os.listdir(root):
    
            filePath = os.path.join(root, file)
    
            if os.path.isdir(filePath):
                i,l = get_image_and_labels_list(root_path, label_path, filePath)
                image_list.extend( i )
                labels_list.extend( l )
    
            else:
                image_list.append(filePath)
                labels_list.append( filePath.replace(root_path, label_path).
                                   replace('_leftImg8bit','_gtFine_labelIds'))
                # Do Stuff
        return image_list, labels_list
    
    