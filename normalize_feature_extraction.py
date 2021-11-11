import pandas as pd
import numpy as np
import os, sys
from scipy.linalg import fractional_matrix_power
import scipy.linalg as la

def load_labels(label_file_path):
	if '.txt' in label_file_path:
		with open(label_file_path, 'r') as label_file:
			lines = label_file.readlines()
			return [l.replace('\n','') for l in lines]
	else:
		return np.load(label_file_path).tolist()

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


def extract_classes(data, classes):
	# return a list of len C with one list of data points per class
	data_classes = [[] for i in range(len(list(set(classes))))]
	for i in range(len(classes)):
		label = classes[i]
		if label == 1:
			data_classes[0].append(data[i])
		else:
			data_classes[1].append(data[i])
	return [np.asarray(data_classes[0]), np.asarray(data_classes[1])]

def transform_for_class(labels, class_label):
	# transform each label of class label to 1 and every other to 0
	new_labels = []
	for i in range(len(labels)):
		current_label = labels[i]
		if int(current_label) == int(class_label):
			new_labels.append(1)
		else:
			new_labels.append(0)
	return new_labels

def normalize_samples(data_set):
	normalized_data = []
	for sample in data_set:
		sample_abs_max = max(abs(np.amax(sample)), abs(np.amin(sample)))
		sample = sample / sample_abs_max
		normalized_data.append(sample)
	return np.asarray(normalized_data)

"""
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

]"""

def apply_normlized_feature_extraction(raw_file, raw_label, raw_save_path, ev_file, ev_save_path, class_label=1):
	save_file_base = f'{raw_save_path}'
	ev_save_file_base = f'{ev_save_path}'

	data_file_path = raw_file
	labels = load_labels(raw_label)
	labels = transform_for_class(labels, class_label)
	data = np.load(data_file_path)
	#data_abs_max = max(abs(np.amax(data)),abs(np.amin(data)))
	#data = data / data_abs_max
	data = normalize_samples(data)
	data_classes = extract_classes(data, labels)

	ev_data_file_path = ev_file
	ev_data = np.load(ev_data_file_path)
	#ev_data_abs_max = max(abs(np.amax(ev_data)),abs(np.amin(ev_data)))
	#ev_data = ev_data / ev_data_abs_max
	ev_data = normalize_samples(ev_data)


	class1_fscores = calculate_Fscores(data_classes[0], data_classes[1], data)

	# calculate average fscore -> TODO discuss this
	class1_fscore_avg = np.mean(class1_fscores)
	# select indices of features above fscore

	class1_f_idx = [i for i in range(0,len(class1_fscores)) if class1_fscores[i] >= class1_fscore_avg]
	
	# save selected feature
	np.save(f"{raw_save_path}_filters.npy", class1_f_idx)

	class1_f_data = []
	ev_class1_f_data = []
	for d in data:
		class1_f_data.append(np.asarray([d[i] for i in class1_f_idx]))
	for d in ev_data:
		ev_class1_f_data.append(np.asarray([d[i] for i in class1_f_idx]))
	
	class1_f_data = np.asarray(class1_f_data)
	ev_class1_f_data = np.asarray(ev_class1_f_data)

	print(class1_f_data.shape)
	np.save(f'{save_file_base}_class{class_label}.npy', class1_f_data)

	print(ev_class1_f_data.shape)
	np.save(f'{ev_save_file_base}_class{class_label}.npy', ev_class1_f_data)


def main(raw_file, raw_label, raw_save_path, ev_file, ev_save_path, class_label=1):
	apply_normlized_feature_extraction(raw_file, raw_label, raw_save_path, ev_file, ev_save_path, class_label)

if __name__ == "__main__":
	main(*sys.argv[1:])