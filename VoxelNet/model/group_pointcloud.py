#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import time

from config import cfg


class VFELayer(object):

    def __init__(self, out_channels, name):
        super(VFELayer, self).__init__()
        self.units = int(out_channels / 2)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            self.dense = tf.layers.Dense(
                self.units, tf.nn.relu, name='dense', _reuse=tf.AUTO_REUSE, _scope=scope)
            self.batch_norm = tf.layers.BatchNormalization(
                name='batch_norm', fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)

    def apply(self, inputs, mask, training):
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        pointwise = self.batch_norm.apply(self.dense.apply(inputs), training)

        #n [K, 1, units]
        aggregated = tf.reduce_max(pointwise, axis=1, keep_dims=True)

        # [K, T, units]
        repeated = tf.tile(aggregated, [1, cfg.VOXEL_POINT_COUNT, 1])

        # [K, T, 2 * units]
        concatenated = tf.concat([pointwise, repeated], axis=2)

        mask = tf.tile(mask, [1, 1, 2 * self.units])

        concatenated = tf.multiply(concatenated, tf.cast(mask, tf.float32))

        return concatenated


class VFENoMaxPoolLayer(object):
    
    def __init__(self, out_channels, name):
        super(VFENoMaxPoolLayer, self).__init__()
        self.units = int(out_channels)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            self.dense = tf.layers.Dense(
                self.units, tf.nn.relu, name='dense', _reuse=tf.AUTO_REUSE, _scope=scope)
            self.batch_norm = tf.layers.BatchNormalization(
                name='batch_norm', fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)

    def apply(self, inputs, mask, training):
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        pointwise = self.batch_norm.apply(self.dense.apply(inputs), training)

        mask = tf.tile(mask, [1, 1, 2 * self.units])

        pointwise = tf.multiply(pointwise, tf.cast(mask, tf.float32))

        return pointwise


class VFENoPointwiseLayer(object):
    
    def __init__(self, out_channels, name):
        super(VFENoPointwiseLayer, self).__init__()
        self.units = int(out_channels)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            self.dense = tf.layers.Dense(
                self.units, tf.nn.relu, name='dense', _reuse=tf.AUTO_REUSE, _scope=scope)
            self.batch_norm = tf.layers.BatchNormalization(
                name='batch_norm', fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)

    def apply(self, inputs, mask, training):
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        pointwise = self.batch_norm.apply(self.dense.apply(inputs), training)

        #n [K, 1, units]
        aggregated = tf.reduce_max(pointwise, axis=1, keep_dims=True)

        # [K, T, units]
        repeated = tf.tile(aggregated, [1, cfg.VOXEL_POINT_COUNT, 1])

        mask = tf.tile(mask, [1, 1, 2 * self.units])

        repeated = tf.multiply(repeated, tf.cast(mask, tf.float32))

        return repeated


class FeatureNet(object):

    def __init__(self, training, batch_size, name='', vfe='VFELayer'):
        super(FeatureNet, self).__init__()
        self.training = training

        # scalar
        self.batch_size = batch_size
        # [ΣK, 35/45, 7]
        self.feature = tf.placeholder(
            tf.float32, [None, cfg.VOXEL_POINT_COUNT, 7], name='feature')
        # [ΣK]
        self.number = tf.placeholder(tf.int64, [None], name='number')
        # [ΣK, 4], each row stores (batch, d, h, w)
        self.coordinate = tf.placeholder(
            tf.int64, [None, 4], name='coordinate')

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            if vfe=='VFELayer':
                self.vfe1 = VFELayer(32, 'VFE-1')
                self.vfe2 = VFELayer(128, 'VFE-2')
            elif vfe=='VFENoMaxPoolLayer':
                self.vfe1 = VFENoMaxPoolLayer(32, 'VFE-1')
                self.vfe2 = VFENoMaxPoolLayer(128, 'VFE-2')
            elif vfe=='VFENoPointwiseLayer':
                self.vfe1 = VFENoPointwiseLayer(32, 'VFE-1')
                self.vfe2 = VFENoPointwiseLayer(128, 'VFE-2')

        # boolean mask [K, T, out_channels]
        mask = tf.not_equal(tf.reduce_max(
            self.feature, axis=2, keep_dims=True), 0)
        x = self.vfe1.apply(self.feature, mask, self.training)
        x = self.vfe2.apply(x, mask, self.training)

        # [ΣK, 128]
        voxelwise = tf.reduce_max(x, axis=1)

        # car: [N * 10 * 400 * 352 * 128]
        # pedestrian/cyclist: [N * 10 * 200 * 240 * 128]
        self.outputs = tf.scatter_nd(
            self.coordinate, voxelwise, [self.batch_size, 10, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128])