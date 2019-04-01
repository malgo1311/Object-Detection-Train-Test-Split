# Object-Detection-Train-Test-Split
When you have image data and corresponding xml files in a single folder, this script will help in splitting the data in two folders - train and test

Usage:

python train_test_split.py \
       \t --datadir='images/all/' \
       \t --split=0.1 \
       \t --train_output='images/train/' \
       \t --test_output='images/test/'
