#!/usr/bin/python3

import os

VEL_TRAIN = './cropped_dataset/training/velodyne/'
VEL_VAL = './cropped_dataset/validation/velodyne/'

#pad each filename with 0's
for filename in os.listdir(VEL_TRAIN):
    new_filename = filename.zfill(10)
    os.rename(VEL_TRAIN+filename, VEL_TRAIN+new_filename)

for filename in os.listdir(VEL_VAL):
    new_filename = filename.zfill(10)
    os.rename(VEL_VAL+filename, VEL_VAL+new_filename)