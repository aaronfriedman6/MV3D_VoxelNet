#!/bin/bash

wget "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip"

for i in 01 17 29 52 70 02 18 35 56 79 19 36 05 57 84 20 39 59 86 11 23 46 60 91 13 27 48 61 15 28 51 64
    do
	wget "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_00${i}/2011_09_26_drive_00${i}_extract.zip"
	wget "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_00${i}/2011_09_26_drive_00${i}_sync.zip"
	wget "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_00${i}/2011_09_26_drive_00${i}_tracklets.zip"
    done
