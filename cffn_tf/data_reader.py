import os
import json
import tensorflow as tf
import numpy as np

train_split = 0.9
label_dict = {'REAL': 1, 'FAKE': 0}

def read_labeled_image_list(image_list_file, training_img_dir):
    # Load the data from the json file
    with open(image_list_file) as f:
        data = json.load(f)

    # the total size of the data
    num_examples = len(data)
    
    # the number of data points to use. To use the whole dataset, set this to
    # num_examples
    num_train_subset = num_examples

    num_train = int(num_train_subset * train_split)

    # Create a list of the all keys and values (for easier slicing and ordering)
    all_imgs = list(data.keys())[:num_train_subset]
    all_lbls = list(data.values())[:num_train_subset]

    # Create the lists of image filenames and labels for both the training set
    # and validation set. For the image filenames, we prepend the path of the
    # image directory. For labels, we convert the strings to integers based on
    # the label_dict.

    train_img = all_imgs[:num_train]
    train_lbl = all_lbls[:num_train]
    val_img = all_imgs[num_train:]
    val_lbl = all_lbls[num_train:]

    train_images = [training_img_dir + i for i in train_img]
    train_labels = [label_dict[lbl] for lbl in train_lbl]

    val_images = [training_img_dir + i for i in val_img]
    val_labels = [label_dict[lbl] for lbl in val_lbl]

    return train_images, train_labels, val_images, val_labels
    
def read_images_from_disk(input_queue, crop_size=64):
    # Read each image from disk and convert to a tensor representation.
    image_filename, label = input_queue
    file_contents = tf.read_file(image_filename)
    tensorized_image = tf.image.decode_jpeg(file_contents, channels=3)

    # Resize the images to the appropriate size.
    tensorized_image = tf.image.resize_images(tensorized_image, [crop_size, crop_size])
    return tensorized_image, label, image_filename
    
def data_queue_helper(images, labels, batch_size, sess, isTrain=True):
    # Only shuffle for training data
    input_queue = tf.train.slice_input_producer([images, labels], shuffle=isTrain)
    image, label, image_filename = read_images_from_disk(input_queue)

    channels = 3
    image.set_shape([None, None, channels])

    # Crop and other random augmentations for the training images
    if isTrain:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_saturation(image, .95, 1.05)
        image = tf.image.random_brightness(image, .05)
        image = tf.image.random_contrast(image, .95, 1.05)
        
    # Apply the standard /255.0 normalization.
    image = tf.cast(image, tf.float32)/255.0
    
    numThreads = 4 if isTrain else 1
    image, label, image_filename = tf.train.batch([image, label, image_filename],
                                                  batch_size=batch_size,
                                                  capacity=batch_size*3,
                                                  num_threads=numThreads,
                                                  name='labels_and_images')

    # Create and return the batched reader
    tf.train.start_queue_runners(sess=sess)

    return image, label

def setup_inputs(sess, filenames, training_img_dir, batch_size=128):
    # Read the Image File List 
    train_images, train_labels, val_images, val_labels = read_labeled_image_list(filenames, training_img_dir)
    num_train = len(train_labels)
    num_val = len(val_labels)

    # Cast the image filenames and training labels to the right type
    train_images = tf.cast(train_images, tf.string)
    train_labels = tf.cast(train_labels, tf.int64)
    val_images = tf.cast(val_images, tf.string)
    val_labels = tf.cast(val_labels, tf.int64)

    # Create the batched reader and return
    train_img, train_lbls = data_queue_helper(train_images, train_labels, batch_size, sess, isTrain=True)
    val_img, val_lbls = data_queue_helper(val_images, val_labels, batch_size, sess, isTrain=False)

    return train_img, train_lbls, num_train, val_img, val_lbls, num_val
