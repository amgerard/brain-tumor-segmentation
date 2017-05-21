from __future__ import division, print_function
#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from skimage.util import img_as_float

import os
from glob import glob
import SimpleITK as sitk

import sys
sys.path.append('/home/amgerard/src/tf_unet')

from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util
from tf_unet import image_util
#from mask_hans import combineMasks
from get_axial_slices import get_data

def create_tensors_from_nii(path):
        img_lbl = combineMasks()
        #imgs = [np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(p)), axis=3) for p in img_paths[0:4]]
        #img_concat = np.concatenate(imgs, axis=3)
        img_concat = sitk.GetArrayFromImage(sitk.ReadImage(path))
        print(img_concat.shape)
        return img_concat, img_lbl


def normalize(slice):
        '''
        INPUT:  (1) a single slice of any given modality (excluding gt)
        (2) index of modality assoc with slice (0=flair, 1=t1, 2=t1c, 3=t2)
        OUTPUT: normalized slice
        '''
        b, t = np.percentile(slice, (1.0,99.0))
        slice = np.clip(slice, b, t)
        if np.std(slice) == 0:
                return slice
        else:
                return (slice - np.mean(slice)) / np.std(slice)

# In[6]:

#x_raw, y_raw = create_tensors_from_nii('/home/amgerard/uiowa/research/nii/ai_msles2_1mm/T1_ai_msles2_1mm_pn0_rf0.nii')
xx,xv,y,yv = get_data(True)

# one-hot
y = y.astype(int)
n_values = np.max(y) + 1
yy = np.eye(n_values)[y]

class AlexDataProvider(image_util.BaseDataProvider):
    """
    Generic data provider for images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix 
    e.g. 'train/fish_1.tif' and 'train/fish_1_mask.tif'
    Usage:
    data_provider = ImageDataProvider("..fishes/train/*.tif")
        
    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param data_suffix: suffix pattern for the data images. Default '.tif'
    :param mask_suffix: suffix pattern for the label images. Default '_mask.tif'
    
    """
    
    n_class = 5
    
    def __init__(self, im, msk, a_min=None, a_max=None, data_suffix=".tif", mask_suffix='_mask.tif'):
        super(AlexDataProvider, self).__init__(a_min, a_max)
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.file_idx = -1
        
        #self.data_files = self._find_data_files(search_path)
    
        #assert len(self.data_files) > 0, "No training files"
        #print("Number of files used: %s" % len(self.data_files))
        
        #img = self._load_file(self.data_files[0])
        #self.channels = 1 if len(img.shape) == 2 else img.shape[-1]
        
        self._3d_img = im
        self._mask = msk
        print(im.shape,msk.shape)
        self.channels = 4
        idxs = range(im.shape[0])
        np.random.shuffle(idxs)
        self.idxs = idxs
        
    def _find_data_files(self, search_path):
        all_files = glob.glob(search_path)
        return [name for name in all_files if not self.mask_suffix in name]
    
    
    def _load_file(self, path, dtype=np.float32):
        return np.array(Image.open(path), dtype)
        # return np.squeeze(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE))

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= self._3d_img.shape[0]:
            self.file_idx = 0
            
    def _load_data_and_label(self):
        data, label = self._next_data()
        #print('qwe',data.shape)
        nx = data.shape[1]
        ny = data.shape[0]
        #print('iop',data.shape,label.shape,nx,ny,self.channels,self.n_class)
        ret_x = data.reshape(1, ny, nx, self.channels)
        ret_y = label.reshape(1, ny, nx, self.n_class)
        #print(ret_x.shape,ret_y.shape)
        return ret_x,ret_y
        
        #return data, label
        
    def _next_data(self):
        self._cylce_file()
        #image_name = self.data_files[self.file_idx]
        #label_name = image_name.replace(self.data_suffix, self.mask_suffix)
        
        #img = self._load_file(image_name, np.float32)
        #label = self._load_file(label_name, np.bool)
    
        #img = np.expand_dims(self._3d_img[self.file_idx,:,:,:], axis=0)
        #label = np.expand_dims(self._mask[self.file_idx,:,:], axis=0)
    
        img = self._3d_img[self.idxs[self.file_idx],:,:]
        label = self._mask[self.idxs[self.file_idx],:,:]
	#print('aaa',img.shape,label.shape)    
        return img,label

print(xx.shape)
print(yy.shape)
print(reduce(lambda x,y: x*y, xx.shape))

generator = AlexDataProvider(xx, yy) # generator.channels, generator.n_class
#net = unet.Unet(channels=4, n_class=5, layers=3, features_root=16, cost='dice_coefficient') #, cost_kwargs={"class_weights":[2,120,10,60,100]})
net = unet.Unet(channels=4, n_class=5, layers=3, features_root=16, cost_kwargs={"class_weights":[2,120,10,60,100]})
#net = unet.Unet(channels=4, n_class=5, layers=3, features_root=16, cost_kwargs={"class_weights":[.01,.2,.2,.2,.2]})
#net.load("./unet_trained_nii/model.cpkt")

#trainer = unet.Trainer(net, optimizer="momentum",opt_kwargs=dict(learning_rate=.00005)) # .000005  # momentum=0.9
trainer = unet.Trainer(net, optimizer="adam", batch_size=20, opt_kwargs=dict(learning_rate=.00001)) # .000005  # momentum=0.9
path = trainer.train(generator, "./unet_trained_nii", training_iters=20, epochs=50, display_step=10,restore=False)

slice_idx, max_cnt = 0, 0
for i in range(y.shape[0]):
	wtf = np.where(y[i,:,:] > 0)
	cnt = len(wtf[0])
	print('wtx',cnt,i)
	if cnt > max_cnt:
		max_cnt = cnt
		slice_idx = i

print('alex',max_cnt,slice_idx)

test = xx # np.expand_dims(xx[slice_idx,:,:,:],axis=0)

prediction = net.predict("./unet_trained_nii/model.cpkt", test)

print(prediction.max())
print('pred shape', prediction.shape)

for i in range(prediction.shape[0]):
	print('idx',i)
	pred_i = np.zeros([5])
	act_i = np.zeros([5])
	for k in range(5):
		pred_i[k] = len(np.where(np.argmax(prediction[i,:,:,:],axis=2) == k)[0])
		act_i[k] = len(np.where(y[i,:,:] == k)[0])
	print(act_i)
	print(pred_i)
		
'''
for i in range(prediction.shape[1]):
    for j in range(prediction.shape[2]):
        #for k in range(4):
            #if (prediction[0,i,j,k] > 0.262):
	    if (np.argmax(prediction[0,i,j,:]) != 0):
		print(np.argmax(prediction[0,i,j,:]), prediction[0,i,j,:])
'''
print(xx[:,:,:,0].shape)


