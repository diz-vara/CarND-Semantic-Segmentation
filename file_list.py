# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 14:33:20 2017

@author: avarfolomeev
"""
import os

def get_image_and_labels_list(root_path, mode, image_path, label_path):
    image_list = []
    label_list = []

    image_mode_dir = os.path.join(root_path, image_path, mode)
    label_mode_dir = os.path.join(root_path, label_path, mode)

    cities = os.listdir(image_mode_dir)
    
    for city in cities:
        image_city_dir = os.path.join(image_mode_dir, city)
        label_city_dir = os.path.join(label_mode_dir, city)
        images = os.listdir(os.path.join(image_city_dir))
        for image_file in images:
            image_list.append(os.path.join(image_city_dir, image_file))
            label_file = image_file.replace('_leftImg8bit','_gtFine_labelIds');
            label_list.append(os.path.join(label_city_dir, label_file))
            
    return image_list, label_list
    
    