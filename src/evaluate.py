import tflearn
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import get_data2

def predict(model, testX):
	pred = np.zeros([testX.shape[0]], np.int8)
	index = 0
	patches_subset = []
	for ix in range(testX.shape[0]):
		#x = np.empty([len(patches_idxs),33,33,4])
		idxs = testX[ix,:]
		im_idx = idxs[0]
		i,j,k = idxs[1:]
		patch = get_data2.all_imgs[im_idx][i,j-16:j+17,k-16:k+17]
		patches_subset.append(patch)
		if len(patches_subset) == 1000 or ix == testX.shape[0]-1:
			pp = np.argmax(model.predict(patches_subset),axis=1)
			for xx in range(len(patches_subset)):
				pred[index] = pp[xx]
				index = index + 1
			patches_subset = []
	return pred

def evaluate_model(model,testX,testY):
	print('accuracy', model.evaluate(testX,testY))
	ty = np.argmax(testY,axis=1)
	#print model.evaluate(testX,testY)
	#tp = np.argmax(model.predict(testX),axis=1)
	tp = predict(model, testX)
	conf_arr = confusion_matrix(ty,tp)
	print(conf_arr)

	norm_conf = []
	for i in conf_arr:
	    a = 0
	    tmp_arr = []
	    a = sum(i, 0)
	    for j in i:
		tmp_arr.append(float(j)/float(a))
	    norm_conf.append(tmp_arr)

	fig = plt.figure()
	plt.clf()
	ax = fig.add_subplot(111)
	ax.set_aspect(1)
	res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
			interpolation='nearest')

	width, height = conf_arr.shape

	for x in xrange(width):
	    for y in xrange(height):
		ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
			    horizontalalignment='center',
			    verticalalignment='center')

	cb = fig.colorbar(res)
	alphabet = ['HT','N','E','NET','ET']
	plt.xticks(range(width), alphabet[:width])
	plt.yticks(range(height), alphabet[:height])
	plt.savefig('confusion_matrix.png', format='png')
