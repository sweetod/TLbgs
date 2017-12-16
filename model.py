# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import time
import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.contrib.slim.python.slim.nets.alexnet import alexnet_v2
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base, inception_v3, inception_v3_arg_scope
from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2_50
from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2_152
from tensorflow.contrib.slim.python.slim.nets.resnet_utils import resnet_arg_scope
import time



height = 1210
width = 1210

sess = tf.InteractiveSession()

batch_size = 20
x = tf.placeholder(tf.float32, shape=[batch_size, height, width, 6])
y_ = tf.placeholder(tf.float32, shape=[batch_size, height, width, 1])

network = tl.layers.InputLayer(x, name='input')

network = tl.layers.Conv2d(network, 16, (3, 3), (1, 1), act=tf.nn.relu, padding='VALID', name='conv1')
network = tl.layers.Conv2d(network, 16, (3, 3), (1, 1), act=tf.nn.relu, padding='VALID', name='conv2')
network = tl.layers.Conv2d(network, 32, (3, 3), (1, 1), act=tf.nn.relu, padding='VALID', name='conv3')
network = tl.layers.MaxPool2d(network, (2, 2), (2, 2), padding='VALID', name='pool1')

network = tl.layers.Conv2d(network, 32, (3, 3), (1, 1), act=tf.nn.relu, padding='VALID', name='conv4')
network = tl.layers.Conv2d(network, 32, (3, 3), (1, 1), act=tf.nn.relu, padding='VALID', name='conv5')
network = tl.layers.Conv2dLayer(network, shape = [1, 1, 32, 3], act=tf.nn.relu, name='conv6', padding='VALID')
network = tl.layers.MaxPool2d(network, (2, 2), (2, 2), padding='VALID', name='pool2')

with slim.arg_scope(resnet_arg_scope()):
    network = tl.layers.SlimNetsLayer(layer=network,
                                  slim_layer=resnet_v2_50,
                                  slim_args={'num_classes':None,
                                             'is_training':True,
                                             'global_pool':False,
                                             'output_stride':16,
                                             # 'inputs' : [batch_size, 299, 299, 3],
                                             },
                                  name='resnet_v2_50'  # <-- the name should be the same with the ckpt model
                                        )
# network = tl.layers.MaxPool2d(network, (1, 1), (1, 1), padding='SAME', name='pool3')
network.print_layers()

network = tf.layers.





