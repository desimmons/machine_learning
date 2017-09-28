def scraper():
	import os
	import scipy.io
	import numpy as np

	envType = ["Anechoic", "Lab", "Reverb"]
	batch = [[],[]]

	for j in envType:
		path = "../Original_data/Data_separated_S1_S2/dBm_in/" + j 
		fileList = os.listdir(path)
		for i in range(0,len(fileList)):
			path = "../Original_data/Data_separated_S1_S2/dBm_in/" + j \
					+"/"+ fileList[i]
			a = scipy.io.loadmat(path)
			if fileList[i][3:5] == "12" or fileList[i][3:5] == "21":
				batch[0].append(a[fileList[i][:-4]][0])
				batch[1].append(1)
			else:
				batch[0].append(a[fileList[i][:-4]][0])
				batch[1].append(0)
			

	snippetLength = 20000
	overlap = 40
	X = []
	Y = []
	for j in range(0,len(batch[0])):
		for i in range(0,len(batch[0][0])-1,snippetLength/overlap):
			if len(batch[0][0]) > i+snippetLength:
				X.append(np.array(batch[0][j][i:i+snippetLength]) \
				 	/np.linalg.norm(batch[0][j][i:i+snippetLength]))
				Y.append(batch[1][j])	
	return X,Y

def feature(X):
	import numpy as np
	import scipy.stats
	features = []
	for i in range(0,len(X)):
		temp = [np.var(X[i]),\
				np.percentile(X[i],10),\
				np.percentile(X[i],20),\
				np.percentile(X[i],30),\
				np.percentile(X[i],40),\
				np.percentile(X[i],50),\
				np.percentile(X[i],60),\
				np.percentile(X[i],70),\
				np.percentile(X[i],80),\
				np.percentile(X[i],90),\
				np.amin(X[i]),\
				np.amax(X[i]),\
				np.ptp(X[i]),\
				scipy.stats.skew(X[i])]
		features.append(temp/np.linalg.norm(temp))
	return np.array(features)