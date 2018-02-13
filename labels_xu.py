#!/usr/bin/python
#
# Cityscapes labels
#

from collections import namedtuple


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

labels_xu = [
    #       name                id    color
    sLabel(  'unlabeled'      ,  0 , (  0,  0,  0) ),
    sLabel(  'sky'            ,  1 , ( 70,130,180) ),
    sLabel(  'road'           ,  2 , (128, 64,128) ),
    sLabel(  'sidewalk'       ,  3 , (244, 35,232) ),
    sLabel(  'Bicycle lane'   ,  4 , (244, 35,132) ),
    sLabel(  'Lane markers'   ,  5 , (255,255,140) ),
    sLabel(  'Railway'        ,  6 , ( 55, 55, 55) ),
    sLabel(  'Grass'          ,  7 , (152,251,152) ),
    sLabel(  'Tree'           ,  8 , (107,142, 35) ),
    sLabel(  'Vegetation'     ,  9 , ( 50,142, 50) ),
    sLabel(  'Building'       , 10 , ( 70, 70, 70) ),
    sLabel(  'Bridge'         , 11 , (150,100,100) ),
    sLabel(  'Pole'           , 12 , (153,153,153) ),
    sLabel(  'Panel'          , 13 , (200,200,  0) ),
    sLabel(  'traffic light'  , 14 , (250,170, 30) ),
    sLabel(  'Fence'          , 15 , (210,153,153) ),
    sLabel(  'Construction'   , 16 , (180,185,180) ),
    sLabel(  'Car'            , 17 , (  0,  0,142) ),
    sLabel(  'Truck'          , 18 , (  0,  0, 70) ),
    sLabel(  'Bus'            , 19 , (  0, 60,100) ),
    sLabel(  'Train'          , 20 , (  0, 80,100) ),
    sLabel(  'Adult'          , 21 , (220, 20, 60) ),
    sLabel(  'Children'       , 22 , (220, 20, 60) ),
    sLabel(  'Cyclist'        , 23 , (255,  0,  0) ),
    sLabel(  'bicycle'        , 24 , (119, 11, 32) ),
    sLabel(  'Motoryclist'    , 25 , (255,  0,  0) ),
    sLabel(  'Motorcycle'     , 26 , (  0,  0,230) ),
    sLabel(  'Anymal'         , 27 , (250, 15, 50) ),
    sLabel(  'Movable'        , 28 , (150, 50,  0) )
]

colors_xu = np.array([label.color for label in labels_xu]).astype(np.uint8)

