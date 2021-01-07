import pandas as pd
import numpy as np
import os, sys
from scipy.linalg import fractional_matrix_power


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

def calculate_trial_R(trial_data):
	single_trial_l1 = trial_data
	single_trial_l1_T = np.transpose(single_trial_l1)
	R_l1_0_nom = single_trial_l1.dot(single_trial_l1_T)
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
			np.add(class_R, trial_R)
	class_R = class_R / len(class_indexes[class_idx])
	return class_R

# EQ_3
def calculate_singular_value_decomposition(R):
	R_eigvals, R_eigenvec = np.linalg.eig(R)
	R_eigvals = R_eigvals.astype(np.float)
	R_U = R_eigenvec.astype(np.float)
	R_lambda = np.diag(R_eigvals)
	return R_U, R_lambda


# EQ_4
def calculate_covariance_transformation_matrix(R_lambda, R_U):
	return fractional_matrix_power(R_lambda, -0.5).dot(np.transpose(R_U)).astype(np.float)

current_wd = os.getcwd()

low_pass = 7
high_pass = 30
file_root = 'BCI4_2a_A01T'
base_path = os.path.join(current_wd, f'Preprocessed\{file_root}_car')
label_file_path = f'{base_path}_labels.txt'
data_file_path = f'{base_path}_{low_pass}_{high_pass}.npy'

labels = load_labels(label_file_path)
data = np.load(data_file_path)
trials = data.shape[0]
time_steps = data.shape[1]
channels = data.shape[2]
class_idxs = extract_class_indexes(labels)
class_count = len(class_idxs)

R_1 = calculate_covariance_for_class(0, class_idxs, data)
R_2 = calculate_covariance_for_class(1, class_idxs, data)
R_3 = calculate_covariance_for_class(2, class_idxs, data)
R_4 = calculate_covariance_for_class(3, class_idxs, data)


R_1_U, R_1_lambda = calculate_singular_value_decomposition(R_1)
P_1 = calculate_covariance_transformation_matrix(R_1_lambda, R_1_U)
R_1_tilde = R_2 + R_3 + R_4

print(R_1_tilde.shape)

G_1 = P_1.dot(R_1).dot(np.transpose(P_1))
G_1_tilde = P_1.dot(R_1_tilde).dot(np.transpose(P_1))

print(G_1.shape, G_1_tilde.shape)