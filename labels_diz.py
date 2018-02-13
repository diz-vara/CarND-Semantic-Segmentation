#!/usr/bin/python
#
# Cityscapes labels
#

from collections import namedtuple
import cv2

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
sLabel = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------
#%%
# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels_diz = [
    #       name                id    color
    sLabel(  'unlabeled'      ,  0 , (  0,  0,  0) ),
    sLabel(  'sky'            ,  1 , ( 70,130,180) ),
    sLabel(  'road'           ,  2 , (128, 64,128) ),
    sLabel(  'sidewalk'       ,  3 , (244, 35,232) ),
    # xu 4 -> 3
    sLabel(  'Lane markers'   ,  5 , (255,255,140) ), #4
    sLabel(  'Railway'        ,  6 , ( 55, 55, 55) ), #5
    sLabel(  'Grass'          ,  7 , (152,251,152) ), #6
    # xu 8 -> 9 sLabel(  'Tree'           ,  8 , (107,142, 35) ),
    sLabel(  'Vegetation'     ,  9 , ( 50,142, 50) ), #7
    sLabel(  'Building'       , 10 , ( 70, 70, 70) ), #8
    sLabel(  'Bridge'         , 11 , (150,100,100) ), #9
    sLabel(  'Construction'   , 12 , (153,153,153) ), #10
    sLabel(  'Panel'          , 13 , (200,200,  0) ), #11
    sLabel(  'traffic light'  , 14 , (250,170, 30) ), #12
    # xu 15 -> 12 sLabel(  'Fence'          , 15 , (210,153,153) ),
    # xu 16 -> 12 sLabel(  'Construction'   , 16 , (180,185,180) ),
    sLabel(  'Transport'      , 17 , (  0,  0,142) ), #13
    # xu 18-> 17 sLabel(  'Truck'          , 18 , (  0,  0, 70) ),
    #xu 19 -> 17 sLabel(  'Bus'            , 19 , (  0, 60,100) ),
    #xu 20 -> 17 sLabel(  'Train'          , 20 , (  0, 80,100) ),
    sLabel(  'Pedestrian'     , 21 , (220, 20, 60) ), #14
    #xu 22 -> 21 sLabel(  'Children'       , 22 , (220, 20, 60) ),
    sLabel(  'Rider'          , 23 , (255,  0,  0) ), #15
    sLabel(  'bicycle'        , 24 , (119, 11, 32) ), #16
    #xu 25 -> 23 sLabel(  'Motoryclist'    , 25 , (255,  0,  0) ),
    #xu 26 -> 24 sLabel(  'Motorcycle'     , 26 , (  0,  0,230) ),
    sLabel(  'Anymal'         , 27 , (250, 15, 50) ), #17
    sLabel(  'Movable'        , 28 , (150, 50,  0) )  #18
]

colors_diz = np.array([label.color for label in labels_diz]).astype(np.uint8)

#%%

def xu2diz(base_dir, in_dir, out_dir):
    _xu2diz = np.array([0, 1, 2, 3, 3, 4, 5, 6, 7, 7, 8 ,9, 10, 11, 12, 10, 10, 
          13, 13, 13, 13, 14, 14, 15, 16, 15, 16, 17, 18]).astype(np.uint8)
    

    in_dir = os.path.join(base_dir, in_dir)
    out_dir = os.path.join(base_dir, out_dir)

    try:
        os.makedirs(out_dir)
    except:
        pass


    print('Loading masks from ' + in_dir)

    
    

    im_files = sorted(os.listdir(in_dir))
    cnt = 0
    end_string = ' from ' + str(len(im_files))
    
    
    
    for label_file in im_files:
        print(str(cnt) + end_string)
        cnt = cnt+1
        label_in = cv2.imread(os.path.join(in_dir, label_file),-1)
        label_out = _xu2diz[label_in]
        cv2.imwrite(os.path.join(out_dir, label_file), label_out)
    
    
    
          
          