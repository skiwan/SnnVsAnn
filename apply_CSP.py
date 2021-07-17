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

def calculate_trial_R(trial_data):
	single_trial_l1 = trial_data
	single_trial_l1_T = np.transpose(single_trial_l1)
	R_l1_0_nom = np.dot(single_trial_l1, single_trial_l1_T)
	R_l1_0_denom = np.trace(R_l1_0_nom)
	R_l1_0 = R_l1_0_nom / R_l1_0_denom
	return R_l1_0

# EQ_1
def calculate_covariance_for_class(class_idx, class_indexes, data):
	class_R = None
	for trial_idx in class_indexes[class_idx]:
		trial_data = data[trial_idx]
		trial_R = calculate_trial_R(trial_data)
		if class_R is None:
			class_R = trial_R
		else:
			class_R = np.add(class_R, trial_R)
	class_R = class_R / len(class_indexes[class_idx])
	return class_R

# EQ_3
def calculate_singular_value_decomposition(*R):
	R_eigvals, R_eigenvec = la.eig(*R)
	sort_ord = np.argsort(R_eigvals)
	sort_ord = sort_ord[::-1]
	R_eigvals = R_eigvals[sort_ord]
	R_U = R_eigenvec[:,sort_ord]
	R_lambda = np.diag(R_eigvals)
	return R_U, R_lambda


# EQ_4
def calculate_covariance_transformation_matrix(R_lambda, R_U):
	return np.dot(np.sqrt(la.inv(R_lambda)),np.transpose(R_U))

# EQ_6
def calculate_Gs(P, R, R_tilde):
	G = np.dot(P, np.dot(R, np.transpose(P)))
	G_tilde = np.dot(P, np.dot(R_tilde, np.transpose(P)))
	return G, G_tilde

# EQ_8
def calculate_I(P,U,R,R_tilde):
	return np.transpose(np.transpose(P).dot(U)).dot(R).dot(np.transpose(P).dot(U)) + np.transpose(np.transpose(P).dot(U)).dot(R_tilde).dot(np.transpose(P).dot(U))
"""
current_wd = os.getcwd()

files = [
['BCI4_2a_A01T','BCI4_2a_A01E']
#,['BCI4_2a_A02T','BCI4_2a_A02E']
#,['BCI4_2a_A03T','BCI4_2a_A03E']
#,'BCI4_2a_A04T' T04 is corrupted due to some technical issues during recording
#,['BCI4_2a_A05T','BCI4_2a_A05E']
#,['BCI4_2a_A06T','BCI4_2a_A06E']
#,['BCI4_2a_A07T','BCI4_2a_A07E']
#,['BCI4_2a_A08T','BCI4_2a_A08E']
#,['BCI4_2a_A09T','BCI4_2a_A09E']
#,['BCI3_3a_k3b' ,'BCI3_3a_k3b']
#,['BCI3_3a_k6b' ,'BCI3_3a_k6b']
#,['BCI3_3a_l1b' ,'BCI3_3a_l1b']

]

configs = [
	[7, 30]
#	,[7,14]
#	,[14,30]
#	,[7,12]
#	,[12,17]
#	,[17,22]
#	,[22,27]
#	,[27,32]
#	,[4,8]
#	,[8,12]
#	,[12,16]
#	,[16,20]
#	,[20,24]
#	,[24,28]
#	,[28,32]
#	,[32,36]
#	,[36,40]
#	,[9,14]
#	,[11,16]
#	,[13,18]
#	,[15,20]
#	,[19,24]
#	,[21,26]
#	,[23,28]
#	,[25,30]
]"""


raw_in='F:\KTH\MA-Thesis\ThesisFolder\SnnVsAnn\sraw\A01T_7_30.npy',
raw_label='F:\KTH\MA-Thesis\ThesisFolder\SnnVsAnn\sraw\A01T.npy',
raw_save='F:\KTH\MA-Thesis\ThesisFolder\SnnVsAnn\sraw\A01T_7_30_CSP.npy',
ev_save='F:\KTH\MA-Thesis\ThesisFolder\SnnVsAnn\sraw\EV_A01T_7_30_CSP.npy',


def apply_csp(raw_input_file, raw_label_file, ev_input_file, ev_label_file, base_save_path, ev_save_path, low_pass, high_pass):
	labels = load_labels(raw_label_file)
	data = np.load(raw_input_file)
	trials = data.shape[0]
	time_steps = data.shape[1]
	channels = data.shape[2]
	class_idxs = extract_class_indexes(labels)
	class_count = len(class_idxs)

	ev_labels = load_labels(ev_label_file)
	ev_data = np.load(ev_input_file)
	ev_class_idxs = extract_class_indexes(ev_labels)
	ev_class_count = len(ev_class_idxs)

	R_s = []
	for x in range(0, class_count):
		R_s.append(calculate_covariance_for_class(x, class_idxs, data))

	R_X = sum(R_s)

	R_s_tilde = []
	for x in range(0, class_count):
		R_s_tilde.append((R_X-R_s[x])/3)

	R_x_U, R_x_lambda = calculate_singular_value_decomposition(R_s[0] + R_s_tilde[0])
	P = calculate_covariance_transformation_matrix(R_x_lambda, R_x_U)

	g_pairs = []
	for x in range(0, class_count):
		G, G_tilde = calculate_Gs(P, R_s[x], R_s_tilde[x])
		g_pairs.append([G, G_tilde])

	final_filters = []
	for x in range(0, class_count):
		G_U, G_lambda = calculate_singular_value_decomposition(g_pairs[x][0])
		#G_tilde_U, G_tilde_lambda = calculate_singular_value_decomposition(G_1_tilde)
		V = np.dot(np.transpose(G_U), P).astype(np.float32)
		final_filters.append(np.concatenate((V[:4],V[-4:])))

	for x in range(0, class_count):
		class_x_filtered = []
		for idx in class_idxs[x]:
			data_sample = data[idx]
			filtered = np.dot(final_filters[x], data_sample)
			class_x_filtered.append(filtered)
		class_x_filtered = np.asarray(class_x_filtered)
		#print(class_x_filtered.shape)
		np.save(f'{base_save_path}_class{x+1}.npy', class_x_filtered)

	for x in range(0, ev_class_count):
		ev_class_x_filtered = []
		for idx in ev_class_idxs[x]:
			data_sample = ev_data[idx]
			filtered = np.dot(final_filters[x], data_sample)
			ev_class_x_filtered.append(filtered)
		ev_class_x_filtered = np.asarray(ev_class_x_filtered)
		#print(class_x_filtered.shape)
		np.save(f'{ev_save_path}_class{x+1}.npy', ev_class_x_filtered)

	# reload all CSP files and concatenate, file is ordered in class 72 per class
	raw = []
	for x in range(0, class_count):
		raw.extend(np.load(f'{base_save_path}_class{x+1}.npy').tolist())
	raw = np.asarray(raw)
	np.save(f'{base_save_path}_full.npy', raw)

	raw = []
	for x in range(0, class_count):
		raw.extend(np.load(f'{ev_save_path}_class{x+1}.npy').tolist())
	raw = np.asarray(raw)
	np.save(f'{ev_save_path}_full.npy', raw)

def main(raw_input_file, raw_label_file, ev_input_file, ev_label_file, base_save_path, ev_save_path, low_pass, high_pass):
	apply_csp(raw_input_file, raw_label_file, ev_input_file, ev_label_file, base_save_path, ev_save_path, low_pass, high_pass)

if __name__ == "__main__":
	main(*sys.argv[1:])