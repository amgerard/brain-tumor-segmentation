from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tensorflow as tf
import numpy as np

from tflearn.layers.merge_ops import merge

import cnn_naive
import cnn_2path
import cnn_3x3
import evaluate

if __name__ == '__main__':
	
	name = 'n2'
	logs_dir = 'logs/' + name + '/'
        cp_dir = 'chpts/' + name + '/'
        rid = name
	
	#with tf.device('/gpu:0'):
	#network = cnn_3x3.get_network()
		#network = cnn_2path.get_network()
	network = cnn_naive.get_network()
	
        model = tflearn.DNN(network, tensorboard_verbose=2, tensorboard_dir=logs_dir, checkpoint_path=cp_dir)
       	model.load('saved_models/' + name + '.tfl')
	#evaluate.evaluate_model(model,testX,testY)

	layer1_var = tflearn.variables.get_layer_variables_by_name('Conv2D_1')
	layer2_var = tflearn.variables.get_layer_variables_by_name('Conv2D_2')
	#inputLayer_var = tflearn.variables.get_layer_variables_by_name('inputLayer')

	#result = tf.matmul(inputLayer_var, layer1_var[0]) + layer1_var[1]

	with model.session.as_default():
		print(tflearn.variables.get_value(layer1_var[0]))
		print(tflearn.variables.get_value(layer1_var[1]))
		print(tflearn.variables.get_value(layer2_var[0]))
		print(tflearn.variables.get_value(layer2_var[1]))

