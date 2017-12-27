import re
import random
import numpy as np
import os.path
import os
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import cv2

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))



def get_image_and_labels_list(root_path, mode, image_path, label_path):
    image_list = []
    label_list = []

    image_mode_dir = os.path.join(root_path, image_path, mode)
    label_mode_dir = os.path.join(root_path, label_path, mode)
    
    print (image_mode_dir)

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
    


def gen_batch_function(data_folder, image_shape, num_classes):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    print("num_classes=",num_classes)
    image_paths, label_paths = get_image_and_labels_list(data_folder, 
                                                         'train',
                                                         'leftImg8bit',
                                                         'gtFine')
    image_nr = len(image_paths)
    print("Image Number = ",image_nr)
    one_hot = np.zeros((num_classes, num_classes), np.int32 )
    for i in range(num_classes): one_hot[i,i]=1

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
                
                augmentation_factor = idx % augmentation_coeff;
                #augmentation - cropping
                crop_factor = augmentation_factor // 2;
                mirror_factor = augmentation_factor % 2;
                
                image = scipy.misc.imread(image_file);
                gt_image = cv2.imread(gt_image_file,-1) #scipy.misc.imread(gt_image_file)*255;
                if crop_factor == 0:
                    # do not crop - use origina image
                    cropped = image;
                    gt_cropped = gt_image;
                else:
                    top = int(image.shape[0]*0.2);
                    left = int(image.shape[1]*0.2);
                    bottom = image.shape[0] - top;
                    right = image.shape[1] - left;
                    if (crop_factor == 1):
                        cropped = image[:bottom, :right, :]
                        gt_cropped = gt_image[:bottom, :right]
                    elif (crop_factor == 2):
                        cropped = image[:bottom, left:, :]
                        gt_cropped = gt_image[:bottom, left:]
                    elif (crop_factor == 3):
                        cropped = image[top:, :right, :]
                        gt_cropped = gt_image[top:, :right]
                    elif (crop_factor == 4):
                        cropped = image[top:, left:, :]
                        gt_cropped = gt_image[top:, left:]
                    elif (crop_factor == 5):
                        #central crop
                        left = left//2;
                        top = top // 2;
                        right = left + right;
                        bottom = bottom + top;
                        cropped = image[top:bottom, left:right, :]
                        gt_cropped = gt_image[top:bottom, left:right]

                
                image = scipy.misc.imresize(cropped, image_shape)
                gt_image = scipy.misc.imresize(gt_cropped, image_shape, 'nearest')

                gt_image[gt_image > 35] = 0
                gt_image[gt_image <  0] = 0
                
                #augmentation - mirroring
                if (mirror_factor != 0):
                    image = np.fliplr(image)
                    gt_image = np.fliplr(gt_image)


                onehot_label = one_hot[gt_image]


                images.append(image)
                gt_images.append(onehot_label)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        road_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        road_segmentation = (road_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        road_mask = np.dot(road_segmentation, np.array([[0, 255, 0, 127]]))
        road_mask = scipy.misc.toimage(road_mask, mode="RGBA")

        other_softmax = im_softmax[0][:, 2].reshape(image_shape[0], image_shape[1])
        other_segmentation = (other_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        other_mask = np.dot(other_segmentation, np.array([[200, 0, 0, 127]]))
        other_mask = scipy.misc.toimage(other_mask, mode="RGBA")


        street_im = scipy.misc.toimage(image)
        street_im.paste(road_mask, box=None, mask=road_mask)
        street_im.paste(other_mask, box=None, mask=other_mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, time.strftime("%Y%m%d_%H%M%S"))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

