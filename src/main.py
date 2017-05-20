from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tensorflow as tf
import numpy as np

from tflearn.layers.merge_ops import merge

import tflearn.data_flow as tfdf
from data_flow_alex import FeedDictFlowAlex
tfdf.FeedDictFlow = FeedDictFlowAlex

import cnn_naive
import cnn_2path
import cnn_3x3
import evaluate
# import train

#X, testX, Y, testY = get_data(fpaths=['data100/X.npy','data100/Xv.npy','data100/Y.npy','data100/Yv.npy'])
from get_data2 import get_data, get_val_data, get_trn_data

if __name__ == '__main__':
	
	name = 'n3'
	logs_dir = 'logs/' + name + '/'
        cp_dir = 'chpts/' + name + '/'
        rid = name

	load_trn,load_val = True,False
	if load_trn:
		X, Y = get_trn_data()
		print('done loading',X.shape,Y.shape)
		print(np.sum(Y, axis=0))
	if load_val:
		testX, testY = get_val_data()
		print('done loading',testX.shape,testY.shape)
		print(np.sum(testY, axis=0))
	
	with tf.device('/gpu:0'):
		#network = cnn_3x3.get_network()
		#network = cnn_2path.get_network()
		network = cnn_naive.get_network(do_dropout=False)
	
        model = tflearn.DNN(network, tensorboard_verbose=2, tensorboard_dir=logs_dir, checkpoint_path=cp_dir)
       	#model.load('n3.tfl')
       	model.load(name + '.tfl')
	evaluate.evaluate_model(model,testX,testY)

        # train
        '''print('start training')
        try:
		model.fit({'input': X}, {'target': Y}, n_epoch=2, # 20
			   #validation_set=({'input': testX}, {'target': testY}),
			   snapshot_step=100, show_metric=True, run_id=rid)
		model.save(name + ".tfl")
	except KeyboardInterrupt:
		# Manually save model
		model.save(name + ".tfl")
'''
