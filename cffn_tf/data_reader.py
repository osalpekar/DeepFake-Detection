import os
import json
import tensorflow as tf
import numpy as np

def read_labeled_image_list(image_list_file, training_img_dir):
    f = open(image_list_file, 'r')
    filenames = []
    labels = []

    for line in f:
        filename, label = line[:-1].split(' ')
        filename = training_img_dir+filename
        filenames.append(filename)
        labels.append(int(label))
        
    return filenames, labels
    
    
def read_images_from_disk(input_queue, size1=64):
    image_filename, label = input_queue
    file_contents = tf.read_file(image_filename)
    tensorized_image = tf.image.decode_jpeg(file_contents, channels=3)
    
    tensorized_image = tf.image.resize_images(tensorized_image, [size1, size1])
    return tensorized_image, label, image_filename
    
def setup_inputs(sess, filenames, training_img_dir, image_size=64, crop_size=64, isTest=False, batch_size=128):
    
    # Read each image file
    image_list, label_list = read_labeled_image_list(filenames, training_img_dir)

    images = tf.cast(image_list, tf.string)
    labels = tf.cast(label_list, tf.int64)
     # Makes an input queue
    if isTest is False:
        isShuffle = True
        numThr = 4
    else:
        isShuffle = False
        numThr = 1
        
    input_queue = tf.train.slice_input_producer([images, labels], shuffle=isShuffle)
    image, label, image_filename = read_images_from_disk(input_queue)

    channels = 3
    image.set_shape([None, None, channels])
        
    # Crop and other random augmentations
    if isTest is False:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_saturation(image, .95, 1.05)
        image = tf.image.random_brightness(image, .05)
        image = tf.image.random_contrast(image, .95, 1.05)
        
    image = tf.cast(image, tf.float32)/255.0
    
    image, label, image_filename = tf.train.batch([image, label, image_filename],
                                                  batch_size=batch_size,
                                                  capacity=batch_size*3,
                                                  num_threads=numThr,
                                                  name='labels_and_images')

    tf.train.start_queue_runners(sess=sess)

    return image, label, len(label_list)
