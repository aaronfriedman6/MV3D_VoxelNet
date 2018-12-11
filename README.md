# Towards Better Point Cloud Feature Encoding in Multi-Input Neural Networks for Autonomous Driving Object Detection

## Structure

This repository has two main folders, mv3d and voxelnet. 

The first, mv3d, is based on MV3D (Multi-View 3D Object Detection Network), an autonomous driving model built with a region proposal network and a fusion network architecture that utilizes successive convolutional layers and residual connections. EDSR was first presented in a paper by Chen et al. that can be found [here](https://arxiv.org/abs/1611.07759). The code in this folder is based off of a TensorFlow implementation of the paper that can be found [here](https://github.com/bostondiditeam/MV3D). The model's primary code and structure can be found in `mv3d/src/mv3d_net.py`.

The second, voxelnet, is based on VoxelNet, a model that focuses on constructing a feature representation of the LIDAR data. VoxelNet was first presented in a paper by Zhou and Tuzel that can be found [here](https://arxiv.org/abs/1711.06396). The code in this folder is based off of a TensorFlow implementation of VoxelNet that can be found [here](https://github.com/qianguih/voxelnet). The model's primary code and structure can be found in the `voxelnet/model` folder.

Entering either folder will display the original GitHub's README file for each project.
