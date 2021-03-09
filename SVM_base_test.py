import os
import numpy as np
from sklearn import svm
import random

current_wd = os.getcwd()

files = [
'BCI4_2a_A01T'
,'BCI4_2a_A02T'
,'BCI4_2a_A03T'
#,'BCI4_2a_A04T' T04 is corrupted due to some technical issues during recording
,'BCI4_2a_A05T'
,'BCI4_2a_A06T'
,'BCI4_2a_A07T'
,'BCI4_2a_A08T'
,'BCI4_2a_A09T'
,'BCI3_3a_k3b'
,'BCI3_3a_k6b'
,'BCI3_3a_l1b'
]

configs = [
	'1','2','6','10','12'
]

classes = [
 'class1'
 ,'class2'
 ,'class3'
 ,'class4'
]

preds = []

for f in files:
	for conf in configs:
		for cla in classes:
			# Classes are ordered in the data file
			file_path = os.path.join(current_wd, f'Normalized_Extracted\{f}_car_{conf}_{cla}.npy')
			dataset = np.load(file_path)
			# flatten for 2d svm
			dataset = dataset.reshape((dataset.shape[0],dataset.shape[1]*dataset.shape[2]))
			labels_c = dataset.shape[0] // len(classes)
			cla_nr = int(cla[-1])
			labels = [1 if (x >= (cla_nr-1)*labels_c and x < (cla_nr-1)*labels_c+labels_c) else 0 for x in range(0, dataset.shape[0])]
			
			# Split data into positive and negative samples
			pos_data = [dataset[i] for i in range(0, dataset.shape[0]) if labels[i] == 1]
			neg_data = [dataset[i] for i in range(0, dataset.shape[0]) if labels[i] == 0]
			# Shuffle Just in Case
			random.shuffle(pos_data)
			random.shuffle(neg_data)


			# split into 229 train and 59 test samples
			# TODO change to fuill train and get eval sets
			x_train = pos_data[:int(len(pos_data)*0.8)] + neg_data[:int(len(neg_data)*0.8)]
			y_train = [1 if i <= int(len(pos_data)*0.8) else 0 for i in range(0, len(x_train))]

			x_test = pos_data[int(len(pos_data)*0.8):] + neg_data[int(len(neg_data)*0.8):]
			y_test = [1 if i <= len(pos_data[int(len(pos_data)*0.8):]) else 0 for i in range(0, len(x_test))]
			
			"""print(dataset.shape)
			print(x_train.shape)
			print(len(y_train), y_train)
			print(x_test.shape)
			print(len(y_test), y_test)"""

			clf = svm.SVC()
			clf.fit(x_train, y_train)

			predicts = clf.predict(x_test)
			c = 0
			for l in range(0,len(y_test)):
				if y_test[l] == predicts[l]:
					c += 1


			preds.append([f, conf, cla, c/len(predicts)])

for p in preds:
	print(p)