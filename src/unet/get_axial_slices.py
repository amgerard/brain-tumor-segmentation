import os
from glob import glob
import sys
import random
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

import sys
sys.path.append('/home/amgerard/src/alex/brain-tumor-segmentation/src')

from read_and_normalize import create_tensors_from_mha

def sample_slices(im_idx,x_c,y_c):
	X_tmp = []
	y_tmp = []

	for i in range(x_c.shape[0]):
		if len(np.where(y_c > 0)) > 0:
			X_tmp.append(x_c[i,:,:,:])
			y_tmp.append(y_c[i,:,:])

	# to numpy
	X_new = np.array(X_tmp)
	y_new = np.array(y_tmp)
	print('test',X_new.shape,y_new.shape)
	return X_new,y_new

all_imgs = []

def get_trn_data(pkl='../data/trn_paths.pkl'):
	trn_paths = load_obj(pkl)
	return get_d(trn_paths)

def get_val_data(pkl='../data/val_paths.pkl'):
	val_paths = load_obj(pkl)
	return get_d(val_paths)

def get_data(use_saved=False):
	if use_saved:
		trn_paths = load_obj('/home/amgerard/uiowa/data/trn_paths.pkl')
		val_paths = load_obj('/home/amgerard/uiowa/data/val_paths.pkl')
	else:
		path = '../data/npy_imgs/'
		paths = [path + y for y in os.listdir(path)]
		random.shuffle(paths)
		trn_paths = paths[:240]
		val_paths = paths[240:]
		save_obj('../data/trn_paths.pkl',trn_paths)
		save_obj('../data/val_paths.pkl',val_paths)
	X,Y = get_d(trn_paths)
	Xv,Yv = get_d(val_paths)
	return X,Xv,Y,Yv

def get_d(paths):
	print('-------------------')
        X,Y = [],[]
        XandY = []
        for pp in paths:
		p = pp.replace('../data','/home/amgerard/uiowa/data')
                #x = np.load(p)
                #y = np.load(p.replace('X','Y').replace('npy_imgs','npy_masks'))
		x,y = save_imgs_as_npy()
                all_imgs.append(x)
                im_idx = 0 # all_imgs.index(x)
                x_p,y_p = sample_slices(im_idx,x,y)
		print(im_idx,p,x.shape)
                if x_p != []:
                        #print(x_p.shape,y_p.shape)
                        #X.append(x_p)
                        #Y.append(y_p)
			XandY.append([x_p,y_p])
		if len(XandY) > 0:
			break

	random.shuffle(XandY)
        X_new = np.concatenate([a[0] for a in XandY],axis=0)
        print('x',X_new.shape)
        y_new = np.concatenate([a[1] for a in XandY],axis=0)
        print('y',y_new.shape)
	return X_new, y_new
        #return train_test_split(X_new, y_new, test_size=0.2, random_state=42)

def save_obj(filename,obj):
	with open(filename,'wb') as fp:
		pickle.dump(obj,fp)

def load_obj(filename):
	with open(filename, 'rb') as fp:
		return pickle.load(fp)

def save_imgs_as_npy():

	path = '/home/amgerard/uiowa/research/BRATS2015_Training/'
        hgg_paths = [path + 'HGG/' + y for y in os.listdir(path + 'HGG')]
        lgg_paths = [path + 'LGG/' + y for y in os.listdir(path + 'LGG')]
        paths = hgg_paths + lgg_paths

	cnt = len(paths)
        for i in range(cnt):
		p = paths[i]
		fn = os.path.basename(p)
                x, y = create_tensors_from_mha(p)
		return x,y
		x_c, y_c = crop(x, y)
		print(i,p,x_c.dtype)

		lggOrHgg = 'HGG' if 'HGG' in p else 'LGG'
		np.save('npy_imgs/X_' + lggOrHgg + '_' + fn + '.npy', x_c.astype(np.float32))
		np.save('npy_masks/Y_' + lggOrHgg + '_' + fn + '.npy', y_c.astype(np.float32))
	
if __name__ == '__main__':
	get_data(True)
	#save_imgs_as_npy()
