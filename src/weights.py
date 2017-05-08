from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

from tflearn.layers.merge_ops import merge

import cnn_naive
import cnn_2path
import cnn_3x3
import evaluate

''' show weights '''
if __name__ == '__main__':
	
	name = 'model9'
	logs_dir = 'logs/' + name + '/'
        cp_dir = 'chpts/' + name + '/'
        rid = name
	
	#with tf.device('/gpu:0'):
	#network = cnn_3x3.get_network()
	#network = cnn_2path.get_network()
	network = cnn_naive.get_network()
	
        model = tflearn.DNN(network, tensorboard_verbose=2, tensorboard_dir=logs_dir, checkpoint_path=cp_dir)
       	model.load('saved_models/' + name + '.tfl')

	layer1_var = tflearn.variables.get_layer_variables_by_name('Conv2D_1')
	#layer2_var = tflearn.variables.get_layer_variables_by_name('Conv2D_2')

	with model.session.as_default():
		arr = tflearn.variables.get_value(layer1_var[0])
		print(tflearn.variables.get_value(layer1_var[0]))
		print(tflearn.variables.get_value(layer1_var[1]))

		# Get each 5x5 filter from the 5x5x1x32 array
		for filter_ in range(arr.shape[3]):
			# Get the 5x5x1 filter:
			extracted_filter = arr[:, :, :, filter_]
			# Get rid of the last dimension (hence get 5x5):
			extracted_filter = np.squeeze(extracted_filter)

			print(extracted_filter.shape)
			# display the filter (might be very small - you can resize the window)
			#scipy.misc.imshow(extracted_filter[:,:,0])
			#plt.imshow(extracted_filter[:,:,2])
			for i in range(32):
				ax = plt.subplot(6,6,i+1)
				ax.xaxis.label.set_visible(False)
				ax.yaxis.label.set_visible(False)
				ax.xaxis.set_ticks([])
				ax.yaxis.set_ticks([])
				plt.imshow(extracted_filter[:,:,i])
			plt.show()
			break

