from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tensorflow as tf
import numpy as np

from tflearn.layers.merge_ops import merge

def get_network():
        # input images
        network = input_data(shape=[None,33,33,4], name='input')

        # local pathway [conv + pool + norm]
        #path1 = conv_2d(network, 32, 7, activation='relu', regularizer="L2", padding='valid')
        path1 = conv_2d(network, 64, 7, activation='relu', regularizer="L2", padding='valid')
        path1 = max_pool_2d(path1, 4, 1, padding='valid')
        path1 = dropout(path1, 0.5)
        #path1 = local_response_normalization(path1)

        #path1 = conv_2d(path1, 32, 3, activation='relu', regularizer="L2", padding='valid')
        path1 = conv_2d(path1, 64, 3, activation='relu', regularizer="L2", padding='valid')
        path1 = max_pool_2d(path1, 2, 1, padding='valid')
        path1 = dropout(path1, 0.5)
        #path1 = local_response_normalization(path1)

        # global pathway
        #path2 = conv_2d(network, 80, 13, activation='relu', regularizer="L2", padding='valid')
        path2 = conv_2d(network, 160, 13, activation='relu', regularizer="L2", padding='valid')
        path2 = dropout(path2, 0.5)
        #path2 = local_response_normalization(path2)

        network = merge([path1,path2],'concat',axis=3)

        network = conv_2d(network, 5, 21, activation='relu', regularizer="L2")
        network = flatten(network, name="flatten")

        # softmax + output layers
        network = fully_connected(network, 5, activation='softmax', name='soft')
        network = regression(network, optimizer='adam', learning_rate=0.00005, # 0.0001
                             loss='categorical_crossentropy', name='target', batch_size=500)
	return network
