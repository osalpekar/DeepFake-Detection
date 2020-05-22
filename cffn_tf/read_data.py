import numpy as np
import os, pdb
import cv2
import numpy as np
import random as rn
import tensorflow as tf
import threading
import time
from sklearn import metrics
import utils

#==========================================================================
#=============Reading data in multithreading manner========================
#==========================================================================
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
    label = input_queue[1]
    fn=input_queue[0]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    
    #example = tf.image.decode_png(file_contents, channels=3, name="dataset_image") # png fo rlfw
    example=tf.image.resize_images(example, [size1,size1])
    return example, label, fn
    
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
    image, y,fn = read_images_from_disk(input_queue)

    channels = 3
    image.set_shape([None, None, channels])
        
    # Crop and other random augmentations
    if isTest is False:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_saturation(image, .95, 1.05)
        image = tf.image.random_brightness(image, .05)
        image = tf.image.random_contrast(image, .95, 1.05)
        
    image = tf.cast(image, tf.float32)/255.0
    
    image, y,fn = tf.train.batch([image, y, fn], batch_size=batch_size, capacity=batch_size*3, num_threads=numThr, name='labels_and_images')

    tf.train.start_queue_runners(sess=sess)

    return image, y, fn, len(label_list)

#==========================================================================
#=============Reading data in multithreading manner========================
#==========================================================================
def read_labeled_image_list2(image_list_file, training_img_dir):
    f = open(image_list_file, 'r')
    filenames = []
    filenames2 = []
    labels = []

    for line in f:
        filename, fn2, label = line[:-1].split(' ')
        filename = training_img_dir+filename
        filenames.append(filename)
        filename = training_img_dir+fn2
        filenames2.append(filename)
        labels.append(int(label))
        
    return filenames, filenames2, labels
    
    
def read_images_from_disk2(input_queue, size1=64):
    label = input_queue[2]
    fn=input_queue[0]
    file_contents = tf.read_file(input_queue[0])
    file_contents2 = tf.read_file(input_queue[1])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    example2 = tf.image.decode_jpeg(file_contents2, channels=3)
    
    #example = tf.image.decode_png(file_contents, channels=3, name="dataset_image") # png fo rlfw
    example=tf.image.resize_images(example, [size1,size1])
    example2=tf.image.resize_images(example2, [size1,size1])
    return example, example2, label, fn
    
def setup_inputs2(sess, filenames, training_img_dir, image_size=64, crop_size=64, isTest=False, batch_size=128):
    
    # Read each image file
    image_list, image2_list, label_list = read_labeled_image_list2(filenames, training_img_dir)

    images = tf.cast(image_list, tf.string)
    images2 = tf.cast(image2_list, tf.string)
    labels = tf.cast(label_list, tf.int64)
     # Makes an input queue
    if isTest is False:
        isShuffle = True
        numThr = 4
    else:
        isShuffle = False
        numThr = 1
        
    input_queue = tf.train.slice_input_producer([images, images2, labels], shuffle=isShuffle)
    image, image2, y,fn = read_images_from_disk2(input_queue)

    channels = 3
    image.set_shape([None, None, channels])
    image2.set_shape([None, None, channels])
        
    # Crop and other random augmentations
    if isTest is False:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_saturation(image, .95, 1.05)
        image = tf.image.random_brightness(image, .05)
        image = tf.image.random_contrast(image, .95, 1.05)
        image2 = tf.image.random_flip_left_right(image2)
        image2 = tf.image.random_saturation(image2, .95, 1.05)
        image2 = tf.image.random_brightness(image2, .05)
        image2 = tf.image.random_contrast(image2, .95, 1.05)
        

    image = tf.cast(image, tf.float32)/255.0
    image2 = tf.cast(image2, tf.float32)/255.0
    
    image, image2, y,fn = tf.train.batch([image, image2, y, fn], batch_size=batch_size, capacity=batch_size*3, num_threads=numThr, name='labels_and_images')

    tf.train.start_queue_runners(sess=sess)

    return image, image2, y, fn, len(label_list)
