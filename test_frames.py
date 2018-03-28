# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:28:39 2018

@author: avarfolomeev
"""

#data_folder='/media/avarfolomeev/storage/Data/Frames/'
data_folder='/media/avarfolomeev/storage/Data/CN/2017_11_08_06_32_41'
image_shape=(576,1024)

road_name = 'Xroad-4'
overlay_name = 'Xoverlay-4'

try:
    os.makedirs(os.path.join(data_folder,road_name))
except:
    pass        

try:
    os.makedirs(os.path.join(data_folder,overlay_name))
except:
    pass        


out_shape = (image_shape[0], image_shape[1], num_classes)

l = glob(os.path.join(data_folder, 'Collected', '*.jpg'))

for im_file in l:
    im_out, mask = segment_file(im_file)
    out_file = im_file.replace('/Collected/','/'+ overlay_name + '/')
    out_file = out_file.replace('.jpg','.png')
    scipy.misc.imsave(out_file, im_out)
    print(out_file)
    out_file = im_file.replace('/Collected/','/' + road_name + '/')
    out_file = out_file.replace('.jpg','.png')
    cv2.imwrite(out_file,mask)
