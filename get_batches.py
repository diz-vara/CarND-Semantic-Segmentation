# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 14:27:29 2018

@author: avarfolomeev
"""

   
def get_batches_fn(batch_size):
    """
    Create batches of training data
    :param batch_size: Batch Size
    :return: Batches of training data
    """
    

    image_nr = len(image_paths)
    augmentation_coeff = (1 + 5) * 2
    total_nr = image_nr #* augmentation_coeff;        
    
    indexes = np.arange(total_nr)
    random.shuffle(indexes)
    
    layer_idx = np.arange(image_shape[0]).reshape(image_shape[0],1)
    component_idx = np.tile(np.arange(image_shape[1]),(image_shape[0],1))
    
    
    for batch_i in range(0, total_nr, batch_size):
        images = []
        gt_images = []
        for i in range(batch_i,batch_i+batch_size):
            if ( i >= total_nr):
                i = i - total_nr; #cycle in case of overflow
            idx = indexes[i]
            image_file = image_paths[idx] # // augmentation_coeff]
            gt_image_file = label_paths[idx] # // augmentation_coeff]
            
            image = scipy.misc.imread(image_file);
            #image = cv2.medianBlur(image,5)
            gt_image = cv2.imread(gt_image_file,-1) #scipy.misc.imread(gt_image_file)*255;
            
            
            assert(image.shape[:2] == gt_image.shape[:2])                

            old_shape = image.shape
            min_scale = max((1.,old_shape[1]/(image_shape[1]*2)) )
            max_scale = min((old_shape[1]/image_shape[1], old_shape[0]/image_shape[0]))
            scale = np.random.rand()*(max_scale-min_scale) + min_scale

            image=cv2.resize(image, dsize=None,fx=1./scale, fy=1./scale)
            gt_image=cv2.resize(gt_image, dsize=None,fx=1./scale, fy=1./scale,
                                interpolation=cv2.INTER_NEAREST)


            new_size = image.shape[:2]
            max_x_shift = np.max(0,new_size[1]-image_shape[1])
            max_y_shift = np.max(0,new_size[0]-image_shape[0])
            
            x_shift = int(np.random.rand()*max_x_shift)
            y_shift = int(np.random.rand()*max_y_shift)
            
            cropped = image[y_shift:y_shift+image_shape[0], 
                            x_shift:x_shift+image_shape[1], :]
            
            gt_cropped = gt_image[y_shift:y_shift+image_shape[0], 
                            x_shift:x_shift+image_shape[1]]

            if (np.random.rand() > 0.5):
                cropped = np.fliplr(cropped)
                gt_cropped = np.fliplr(gt_cropped)
                

            gt_cropped[gt_cropped >= num_classes] = 0
            gt_cropped[gt_cropped <  0] = 0
            
            #augmentation - mirroring


            onehot_label = one_hot[gt_cropped]

            #print(image.shape, gt_image.shape)

            images.append(cropped)
            gt_images.append(onehot_label)
            

        yield np.array(images), np.array(gt_images)
            