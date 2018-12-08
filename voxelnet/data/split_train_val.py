# !/usr/bin/python3

import os

IMG_ROOT = './cropped_dataset/training/image_2/'
LAB_ROOT = './cropped_dataset/training/label_2/'
VEL_ROOT = './cropped_dataset/training/velodyne/'
CALIB_ROOT = './calib/training/calib/'
IMG_VAL = './cropped_dataset/validation/image_2/'
LAB_VAL = './cropped_dataset/validation/label_2/'
VEL_VAL = './cropped_dataset/validation/velodyne/'
CALIB_VAL = './calib/validation/calib/'

#open val.txt and put all the file integers into a set
val_set = set()
with open("val.txt","r") as val_file:
    val_num = val_file.readline()
    while val_num != '':
        val_set.add(int(val_num.rstrip("\n")))
        val_num = val_file.readline()

#move val data from training to validation
for frame in range(0, 7481):
    if frame in val_set:
        #training data
        #img_r = IMG_ROOT + '%06d.png' % frame
        #lab_r = LAB_ROOT + '%06d.txt' % frame
        #vel_r = VEL_ROOT + str(frame) + '.bin'
        calib_r = CALIB_ROOT + '%06d.txt' % frame        

        #validation data
        #img_v = IMG_VAL + '%06d.png' % frame
        #lab_v = LAB_VAL + '%06d.txt' % frame
        #vel_v = VEL_VAL + str(frame) + '.bin'
        calib_v = CALIB_VAL + '%06d.txt' % frame

        #os.rename(img_r, img_v)
        #os.rename(lab_r, lab_v)
        #os.rename(vel_r, vel_v)
        os.rename(calib_r, calib_v)
