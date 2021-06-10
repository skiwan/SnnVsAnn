import pandas as pd
import numpy as np
import os, sys
from scipy.linalg import fractional_matrix_power
import scipy.linalg as la

def load_labels(label_file_path):
	with open(label_file_path, 'r') as label_file:
		lines = label_file.readlines()
		return [l.replace('\n','') for l in lines]

def extract_class_indexes(labels):
	label_count = len(list(set(labels)))
	class_idxs = []
	for i in range(0, label_count):
		class_idxs.append([])
	for idx in range(0, len(labels)):
		class_idxs[int(labels[idx])-1].append(idx)
	return class_idxs

def calculate_mean_values_for_subset(subset):
	overall_mean_values = [0 for x in range(0,subset.shape[1])]
	for trial in range(0, subset.shape[0]):
		for channel in range(0, subset.shape[1]):
			for sample in range(0, subset.shape[2]):
				overall_mean_values[channel] += subset[trial][channel][sample]
	for channel in range(0, len(overall_mean_values)):
		overall_mean_values[channel] = overall_mean_values[channel] / (subset.shape[0] * subset.shape[2])
	return overall_mean_values

def calculate_denominator_array(sub_class, sub_class_means):
	denom = [0 for x in range(0,sub_class.shape[1])]
	for channel in range(0, sub_class.shape[1]):
		for trial in range(0, sub_class.shape[0]):
			for sample in range(0, sub_class.shape[2]):
				denom[channel] += (sub_class[trial][channel][sample] - sub_class_means[channel])**2
	return denom


def calculate_Fscores(positive_class, negative_class, whole_set):
	f_scores = [0 for x in range(0,whole_set.shape[1])]
	overall_mean_values = calculate_mean_values_for_subset(whole_set)
	positive_mean_values = calculate_mean_values_for_subset(positive_class)
	negative_mean_values = calculate_mean_values_for_subset(negative_class)

	denom_1 = calculate_denominator_array(positive_class, positive_mean_values)
	denom_2 = calculate_denominator_array(negative_class, negative_mean_values)
	for i in range(0, len(f_scores)):
		nom_1 = (positive_mean_values[i] - overall_mean_values[i])**2
		nom_2 = (negative_mean_values[i] - overall_mean_values[i])**2
		den_1 = 1 / (positive_class.shape[0] * positive_class.shape[2])
		den_2 = 1 / (negative_class.shape[0] * negative_class.shape[2])
		f_scores[i] = (nom_1 + nom_2) / ((den_1 * denom_1[i]) + (den_2 * denom_2[i]))
	return f_scores

current_wd = os.getcwd()

files = [
 ['BCI4_2a_A01T', 'BCI4_2a_A01E']
,['BCI4_2a_A02T', 'BCI4_2a_A02E']
,['BCI4_2a_A03T', 'BCI4_2a_A03E']
,['BCI4_2a_A05T', 'BCI4_2a_A05E']
,['BCI4_2a_A06T', 'BCI4_2a_A06E']
,['BCI4_2a_A07T', 'BCI4_2a_A07E']
,['BCI4_2a_A08T', 'BCI4_2a_A08E']
,['BCI4_2a_A09T', 'BCI4_2a_A09E']
,['BCI3_3a_k3b', 'BCI3_3a_k3b']
,['BCI3_3a_k6b', 'BCI3_3a_k6b']
,['BCI3_3a_l1b', 'BCI3_3a_l1b']
]

configs = [
	[[7, 30]]
	,[[7,14],[14,30]]
	,[[7,12],[12,17],[17,22],[22,27],[27,32],[7,30]]
	,[[4,8],[8,12],[12,16],[16,20],[20,24],[24,28],[28,32],[32,36],[36,40],[7,30]]
	,[[7,12],[9,14],[11,16],[13,18],[15,20],[17,22],[19,24],[21,26],[23,28],[25,30],[27,32],[7,30]]
]

classes = [
 'class1'
 ,'class2'
 ,'class3'
 ,'class4'

]

for f_e in files:
	for conf in configs:
		config_nr = len(conf)
		file_root = f_e[0]
		base_path = os.path.join(current_wd, f'Filtered\{file_root}_car')
		label_file_path = f'Preprocessed\{file_root}_car_labels.txt'
		save_file_base = f'Normalized_Extracted\{file_root}_car_{config_nr}'

		ev_file_root = f_e[1]
		ev_base_path = os.path.join(current_wd, f'Filtered\{ev_file_root}_car')
		ev_label_file_path = f'Preprocessed\{ev_base_path}_car_labels.txt'
		ev_save_file_base = f'Normalized_Extracted\{ev_base_path}_car_{config_nr}'



		labels = load_labels(label_file_path)
		data_classes = []
		
		ev_labels = load_labels(ev_label_file_path)
		ev_data_classes = []

		print(f'start for {file_root}_car_{config_nr}')

		for cla in classes:
			data = None
			ev_data = None
			for l_h in conf:
				low = l_h[0]
				high = l_h[1]
				data_file_path = f'{base_path}_{low}_{high}_{cla}.npy'
				ev_data_file_path = f'{ev_base_path}_{low}_{high}_{cla}.npy'
				
				if data is None:
					data = np.load(data_file_path)
				else:
					data = np.concatenate((data, np.load(data_file_path)), axis=1)

				if ev_data is None:
					ev_data = np.load(ev_data_file_path)
				else:
					ev_data = np.concatenate((ev_data, np.load(ev_data_file_path)), axis=1)
				# uniformly normalize
			data_abs_max = max(abs(np.amax(data)),abs(np.amin(data)))
			data = data / data_abs_max
			data_classes.append(data)

			ev_data_abs_max = max(abs(np.amax(ev_data)),abs(np.amin(ev_data)))
			ev_data = ev_data / ev_data_abs_max
			ev_data_classes.append(ev_data)
				

		whole_set = np.concatenate(data_classes, axis=0)
		ev_whole_set = np.concatenate(ev_data_classes, axis=0)

		class1_fscores = calculate_Fscores(data_classes[0], np.concatenate(data_classes[1:], axis=0), whole_set)

		class2_neg_subset =  np.concatenate(data_classes[2:], axis=0)
		class2_neg_subset =  np.concatenate((data_classes[0],class2_neg_subset), axis=0)
		class2_fscores = calculate_Fscores(data_classes[1], class2_neg_subset, whole_set)
		
		class3_neg_subset =  np.concatenate(data_classes[:2],  axis=0)
		class3_neg_subset =  np.concatenate((class3_neg_subset,data_classes[3]), axis=0)
		class3_fscores = calculate_Fscores(data_classes[2], class3_neg_subset, whole_set)

		class4_fscores = calculate_Fscores(data_classes[3], np.concatenate(data_classes[:3], axis=0), whole_set)

		# calculate average fscore -> TODO discuss this
		class1_fscore_avg = np.average(class1_fscores)
		class2_fscore_avg = np.average(class2_fscores)
		class3_fscore_avg = np.average(class3_fscores)
		class4_fscore_avg = np.average(class4_fscores)
		# select indices of features above fscore

		class1_f_idx = [i for i in range(0,len(class1_fscores)) if class1_fscores[i] >= class1_fscore_avg]
		class2_f_idx = [i for i in range(0,len(class2_fscores)) if class2_fscores[i] >= class2_fscore_avg]
		class3_f_idx = [i for i in range(0,len(class3_fscores)) if class3_fscores[i] >= class3_fscore_avg]
		class4_f_idx = [i for i in range(0,len(class4_fscores)) if class4_fscores[i] >= class4_fscore_avg]
		
		# save selected feature
		#TODO SAVE EV

		class1_f_data = []
		ev_class1_f_data = []
		for data in whole_set:
			class1_f_data.append(np.asarray([data[i] for i in class1_f_idx]))
		
		for data in ev_whole_set:
			ev_class1_f_data.append(np.asarray([data[i] for i in class1_f_idx]))
		
		class1_f_data = np.asarray(class1_f_data)
		ev_class1_f_data = np.asarray(ev_class1_f_data)

		class2_f_data = []
		ev_class2_f_data = []
		for data in whole_set:
			class2_f_data.append(np.asarray([data[i] for i in class2_f_idx]))

		for data in ev_whole_set:
			ev_class2_f_data.append(np.asarray([data[i] for i in class2_f_idx]))
		
		class2_f_data = np.asarray(class2_f_data)
		ev_class2_f_data = np.asarray(ev_class2_f_data)

		class3_f_data = []
		ev_class3_f_data = []
		for data in whole_set:
			class3_f_data.append(np.asarray([data[i] for i in class3_f_idx]))

		for data in ev_whole_set:
			ev_class3_f_data.append(np.asarray([data[i] for i in class3_f_idx]))

		class3_f_data = np.asarray(class3_f_data)
		ev_class3_f_data = np.asarray(ev_class3_f_data)


		class4_f_data = []
		ev_class4_f_data = []

		for data in whole_set:
			class4_f_data.append(np.asarray([data[i] for i in class4_f_idx]))

		for data in ev_whole_set:
			ev_class4_f_data.append(np.asarray([data[i] for i in class4_f_idx]))

		class4_f_data = np.asarray(class4_f_data)
		ev_class4_f_data = np.asarray(ev_class4_f_data)	

		print(class1_f_data.shape, class2_f_data.shape, class3_f_data.shape, class4_f_data.shape)
		np.save(f'{save_file_base}_class1.npy', class1_f_data)
		np.save(f'{save_file_base}_class2.npy', class2_f_data)
		np.save(f'{save_file_base}_class3.npy', class3_f_data)
		np.save(f'{save_file_base}_class4.npy', class4_f_data)

		print(ev_class1_f_data.shape, ev_class2_f_data.shape, ev_class3_f_data.shape, ev_class4_f_data.shape)
		np.save(f'{ev_save_file_base}_class1.npy', ev_class1_f_data)
		np.save(f'{ev_save_file_base}_class2.npy', ev_class2_f_data)
		np.save(f'{ev_save_file_base}_class3.npy', ev_class3_f_data)
		np.save(f'{ev_save_file_base}_class4.npy', ev_class4_f_data)


			


