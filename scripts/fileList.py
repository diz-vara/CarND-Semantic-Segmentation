# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 14:33:20 2017

@author: avarfolomeev
"""


def getFileList(path):
    image_list = []
    labels_list = []
    for root, subdirs, files in os.walk(path):
    
        for file in os.listdir(root):
    
            filePath = os.path.join(root, file)
    
            if os.path.isdir(filePath):
                i,l = getFileList(filePath)
                image_list.extend( i )
                labels_list.extend( l )
    
            else:
                l.append(filePath)
                # Do Stuff
        return l
        
        