# Copyright 2019. All Rights Reserved.
#
# Prepared by: Aishwarya Malgonde
# Date & Time: 5th March 2019 | 12:17:00
# ==============================================================================

r"""Test Train Split.

This executable is used to split train and test datasets. 

Example usage:
    python train_test_split.py \
        --datadir='' \
        --split=0.1 \
        --train_output='images/train/' \
        --test_output='images/test/'

"""

import tensorflow as tf
import os
from random import shuffle
import pandas as pd
from math import floor
import cv2
import xml.etree.ElementTree as ET 

flags = tf.app.flags
flags.DEFINE_string('datadir', '', 'Path to the all input data')
flags.DEFINE_float('split', 0.9, 'Split value - Train %')
flags.DEFINE_string('train_output', '', 'Path to output train data')
flags.DEFINE_string('test_output', '', 'Path to output train data')
FLAGS = flags.FLAGS

def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print('Creating directory -', directory)
    else:
        print('Directory exists -', directory)

def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda file: file.endswith('.jpg'), all_files))
    shuffled_files = randomize_files(data_files)
    all_cervix_images = pd.DataFrame({'imagepath': shuffled_files})
    all_cervix_images['filename'] = all_cervix_images.apply(lambda row: row.imagepath.split(".")[0], axis=1)
    return all_cervix_images

def randomize_files(file_list):
    shuffle(file_list)
    return  file_list

def get_training_and_testing_sets(file_list, split):
    split_index = floor(file_list.shape[0] * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    testing = testing.reset_index(drop=True)
    return training, testing

def write_data(training, testing, datadir, train_output, test_output):
    
    # Train Data
    print ('Writing -', training.shape[0], '- Train data images at -', train_output)
    for name in training['filename']:
        # Reading images
        rd_path = os.path.join(datadir, name+'.jpg')
        image = cv2.imread(rd_path)
        
        # Writing images
        wr_path = os.path.join(train_output, name+'.jpg')
        cv2.imwrite(wr_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
        # Reading xml
        rd_path = os.path.join(datadir, name+'.xml')
        tree = ET.parse(rd_path)
        
        # Writing xml
        wr_path = os.path.join(train_output, name+'.xml')
        tree.write(wr_path)

    # Test Data
    print ('Writing -', testing.shape[0], '- Test data images at -', test_output)
    for name in testing['filename']:
        # Reading images
        rd_path = os.path.join(datadir, name+'.jpg')
        image = cv2.imread(rd_path)
        
        # Writing images
        wr_path = os.path.join(test_output, name+'.jpg')
        cv2.imwrite(wr_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
        # Reading xml
        rd_path = os.path.join(datadir, name+'.xml')
        tree = ET.parse(rd_path)
        
        # Writing xml
        wr_path = os.path.join(test_output, name+'.xml')
        tree.write(wr_path)

def main(_):
    check_dir(FLAGS.train_output)
    check_dir(FLAGS.test_output)
    file_list = get_file_list_from_dir(FLAGS.datadir)
    print('Read -', file_list.shape[0], '- files from the directory -', FLAGS.datadir)
    training, testing = get_training_and_testing_sets(file_list, FLAGS.split)
    write_data(training, testing, FLAGS.datadir, FLAGS.train_output, FLAGS.test_output)

if __name__ == '__main__':
  tf.app.run()
