from PIL import Image
import SimpleITK as sitk
from skimage.util import img_as_float
import numpy as np

def get_color_from_lbl(lbl):
	if lbl == 1: return (0,0,255,0)
	if lbl == 2: return (0,255,255,0)
	if lbl == 3: return (0,255,0,0)
	if lbl == 4: return (255,0,0,0)

A = sitk.GetArrayFromImage(sitk.ReadImage('/Users/amgerard/uiowa/machine_learning/project/BRATS2015_Training/HGG/brats_tcia_pat499_0001/VSD.Brain.XX.O.MR_Flair.36620/VSD.Brain.XX.O.MR_Flair.36620.mha'))
#M = sitk.GetArrayFromImage(sitk.ReadImage('/Users/amgerard/uiowa/machine_learning/project/BRATS2015_Training/HGG/brats_tcia_pat499_0001/VSD.Brain_3more.XX.O.OT.42835/VSD.Brain_3more.XX.O.OT.42835.mha'))
M = np.load('A_HGG_brats_tcia_pat499_0001.npy')

'''A = sitk.GetArrayFromImage(sitk.ReadImage('/Users/amgerard/uiowa/machine_learning/project/BRATS2015_Training/HGG/brats_tcia_pat399_0417/VSD.Brain.XX.O.MR_Flair.36205/VSD.Brain.XX.O.MR_Flair.36205.mha'))
M = sitk.GetArrayFromImage(sitk.ReadImage('/Users/amgerard/uiowa/machine_learning/project/BRATS2015_Training/HGG/brats_tcia_pat399_0417/VSD.Brain_3more.XX.O.OT.42683/VSD.Brain_3more.XX.O.OT.42683.mha'))
M = np.load('A_HGG_brats_tcia_pat399_0417.npy')
'''

print(M.shape)
print(A.shape)
print(A.min(),A.max())
A = (A/float(A.max()))*255.0

maxIdx = 135

if True:
	for i in range(maxIdx):
		im = Image.fromarray(A[i,:,:]).convert('RGB')
		mask_idxs = np.where(M[i,:,:] > 0)
		pxls = im.load()
		for j in range(len(mask_idxs[0])):
			x,y = mask_idxs[0][j],mask_idxs[1][j]
			pxls[x,y] = get_color_from_lbl(M[i,x,y])	
		fn = str(i) + '.png'
		print(i, fn)
		im.save('gif_alex2/' + fn)

	for i in range(maxIdx):
		idx = maxIdx - i - 1
		im = Image.fromarray(A[idx,:,:]).convert('RGB')
		pxls = im.load()
		mask_idxs = np.where(M[i,:,:] > 0)
		for j in range(len(mask_idxs[0])):
			x,y = mask_idxs[0][j],mask_idxs[1][j]
			pxls[x,y] = get_color_from_lbl(M[i,x,y])	
		fn = str(maxIdx + i) + '.png'
		print(idx, fn)
		im.save('gif_alex2/' + fn)

