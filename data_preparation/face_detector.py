import face_recognition
import cv2
from glob import glob
from PIL import Image
import numpy as np
import os

batch_size = 128

img_files = glob("/home/ubuntu/big_data/*.jpg")
dir_name = "/home/ubuntu/face_cropped/"

for start_index in range(0, len(img_files), batch_size):
    # get a batch of image filenames from the directory
    img_batch = img_files[start_index:start_index + batch_size]
    # convert all those image files to numpy arrays
    images = {img: face_recognition.load_image_file(img) for img in img_batch}
    images = {fname: images[fname] for fname in images if images[fname].shape == (1920, 1080, 3)}
    fnames = list(images.keys())
    images = list(images.values())
    # use face_detection API to get face locations
    batch_of_face_locations = face_recognition.batch_face_locations(images, number_of_times_to_upsample=0, batch_size=len(images))

    for i in range(len(batch_of_face_locations)):
        face_location = batch_of_face_locations[i][0]
        # unpack bounding box coordinates of face_locations
        top, right, bottom, left = face_location
        # crop the image to only the face
        face_image = images[i][top:bottom, left:right]
        # generate new filename with original basename in new directory
        new_file_name = dir_name + os.path.basename(fnames[i])
        # save the image file to disk
        img = Image.fromarray(face_image, 'RGB')
        img.save(new_file_name)
    print("a completed batch at position: " + str(start_index))
