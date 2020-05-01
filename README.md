# Object-Detection-Train-Test-Split
When you have image data ('.jpg' format) and corresponding xml files in a single folder, this script will help in splitting the data in two folders - train and test.

The image and corresponding xml file should have same names.

Usage:

python test_train_split.py \
              --datadir='images/all/' \
              --split=0.1 \
              --train_output='images/train/' \
              --test_output='images/test/' \
              --image_ext='jpeg'
