from PIL import Image
import SimpleITK as sitk
from skimage.util import img_as_float
import numpy as np
from glob import glob
import os
def get_color_from_lbl(lbl):
	if lbl == 1: return (0,0,255,0)
	if lbl == 2: return (0,255,255,0)
	if lbl == 3: return (0,255,0,0)
	if lbl == 4: return (255,0,0,0)

# VSD.Brain.XX.O.MR_T1c.36622/VSD.Brain.XX.O.MR_T1c.36622.mha'))
# VSD.Brain_3more.XX.O.OT.42835/VSD.Brain_3more.XX.O.OT.42835.mha'))

path = '/Users/amgerard/uiowa/machine_learning/project/BRATS2015_Training/HGG/brats_tcia_pat499_0001/'

i = 70

flr, msk = '', ''

for p in [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.mha'))]:
	if 'Flair' in p: flr = p
	if 'OT' in p: msk = p
	continue

	A = sitk.GetArrayFromImage(sitk.ReadImage(p))
	print(A.shape)
	print(A.min(),A.max())
	A = (A/float(A.max()))*255.0
	im = Image.fromarray(A[i,:,:]).convert('RGB')
	'''
	mask_idxs = np.where(M[i,:,:] > 0)
	pxls = im.load()
	for j in range(len(mask_idxs[0])):
		x,y = mask_idxs[0][j],mask_idxs[1][j]
		pxls[x,y] = get_color_from_lbl(M[i,x,y])'''
	fn = os.path.splitext(os.path.basename(p))[0] + '.png'
	print(i, fn)
	im.save('imgs/' + fn)

A = sitk.GetArrayFromImage(sitk.ReadImage(flr))
M = sitk.GetArrayFromImage(sitk.ReadImage(msk))
A = (A/float(A.max()))*255.0
im = Image.fromarray(A[i,:,:]).convert('RGB')
mask_idxs = np.where(M[i,:,:] > 0)
pxls = im.load()
for j in range(len(mask_idxs[0])):
	x,y = mask_idxs[0][j],mask_idxs[1][j]
	pxls[x,y] = get_color_from_lbl(M[i,x,y])
fn = os.path.splitext(os.path.basename(flr))[0] + '_truth.png'
print(i, fn)
im.save('imgs/' + fn)
