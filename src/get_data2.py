import os
from glob import glob
import sys
import random
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

def crop(x, y):
	idxsa = np.where(y > 0)
	min0 = idxsa[0].min()-10
	max0 = idxsa[0].max()+10
	min1 = idxsa[1].min()-10
	max1 = idxsa[1].max()+10
	min2 = idxsa[2].min()-10
	max2 = idxsa[2].max()+10
	x_crop = x[min0:max0,min1:max1,min2:max2]
	y_crop = y[min0:max0,min1:max1,min2:max2]
	return x_crop, y_crop

def is_in_bounds(im, i, j, k):
	return j-16 >= 0 and j+17 < im.shape[1] and k-16 >= 0 and k+17 < im.shape[2]

def sample_patches(im_idx,x_c,y_c):
	X_tmp = []
	y_tmp = []
	for lbl in range(5): # each label
		lbl_idxs = np.where(y_c == lbl)
		lbl_cnt = len(lbl_idxs[0])
		
		#skip = 1 if lbl_cnt < 500 else 2 if lbl_cnt < 2000 else 5 if lbl_cnt < 5000 else 10
		skip = 1 if lbl_cnt < 8000 else int(lbl_cnt/4000.)
		#print('skip',skip,lbl_cnt)
		for idx in range(lbl_cnt): # each idx for label
			i,j,k = lbl_idxs[0][idx], lbl_idxs[1][idx],lbl_idxs[2][idx]
			if idx % skip == 0 and is_in_bounds(x_c,i,j,k): # is valid
				X_tmp.append([im_idx,i,j,k])
				y_tmp.append(lbl)
	# to numpy
	X_new = np.array(X_tmp)
	y = np.array(y_tmp)

	# one-hot
	y_new = np.zeros([y.shape[0],5])
	try:
		y_new[np.arange(y.shape[0]),y] = 1
	except:
		print('WTF',im_idx)
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
		trn_paths = load_obj('../data/trn_paths.pkl')
		val_paths = load_obj('../data/val_paths.pkl')
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
        for p in paths:
                x = np.load(p)
                y = np.load(p.replace('X','Y').replace('npy_imgs','npy_masks'))
                all_imgs.append(x)
                im_idx = all_imgs.index(x)
                x_p,y_p = sample_patches(im_idx,x,y)
		print(im_idx,p,x.shape)
                if x_p != []:
                        #print(x_p.shape,y_p.shape)
                        #X.append(x_p)
                        #Y.append(y_p)
			XandY.append([x_p,y_p])
		#if len(XandY) > 10:
		#	break

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
		x_c, y_c = crop(x, y)
		print(i,p,x_c.dtype)

		lggOrHgg = 'HGG' if 'HGG' in p else 'LGG'
		np.save('npy_imgs/X_' + lggOrHgg + '_' + fn + '.npy', x_c.astype(np.float32))
		np.save('npy_masks/Y_' + lggOrHgg + '_' + fn + '.npy', y_c.astype(np.float32))
	
if __name__ == '__main__':
	get_data(True)
	#save_imgs_as_npy()
