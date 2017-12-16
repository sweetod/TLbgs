#! /usr/bin/env python
# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorlayer as tl

slim = tf.contrib.slim
from tensorflow.contrib.slim.python.slim.nets.resnet_utils import resnet_arg_scope
from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2_50, resnet_v2_101, resnet_v2_152

def slim_resnet_v2_50(net_in,
                      num_classes=1000,
                      weight_decay=0.0001,
                      is_training=False,
                      reuse=None):
    net_in = tl.layers.InputLayer(net_in, name='input_layer')
    with slim.arg_scope(resnet_arg_scope(is_training=is_training, weight_decay=weight_decay)):
        network = tl.layers.SlimNetsLayer(layer=net_in, slim_layer=resnet_v2_50,
                                          slim_args={
                                              'num_classes': num_classes,},
                                          name='resnet_v2_50') # same with the ckpt mode
    return network
slim_resnet_v2_50.default_image_size = 224
slim_resnet_v2_50.default_weight = 'resnet_v2_50.ckpt'
slim_resnet_v2_50.default_exclude_variables = ['resnet_v2_50/logits']