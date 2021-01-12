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
R_X = R_1 + R_2 + R_3 + R_4

R_1_tilde = (R_2 + R_3 + R_4) / 3
R_2_tilde = (R_1 + R_3 + R_4) / 3
R_3_tilde = (R_2 + R_1 + R_4) / 3
R_4_tilde = (R_2 + R_3 + R_1) / 3

R_x_U, R_x_lambda = calculate_singular_value_decomposition(R_1 + R_1_tilde)
P = calculate_covariance_transformation_matrix(R_x_lambda, R_x_U)

G_1, G_1_tilde = calculate_Gs(P, R_1, R_1_tilde)
G_2, G_2_tilde = calculate_Gs(P, R_2, R_2_tilde)
G_3, G_3_tilde = calculate_Gs(P, R_3, R_3_tilde)
G_4, G_4_tilde = calculate_Gs(P, R_4, R_4_tilde)

# PROBLEM ! G_1_U and G_1_tilde_U are not the same even thou
# They are supposed to be
G_1_U, G_1_lambda = calculate_singular_value_decomposition(G_1)
G_1_tilde_U, G_1_tilde_lambda = calculate_singular_value_decomposition(G_1_tilde)


G_2_U, G_2_lambda = calculate_singular_value_decomposition(G_2)
G_2_tilde_U, G_2_tilde_lambda = calculate_singular_value_decomposition(G_2_tilde)

G_3_U, G_3_lambda = calculate_singular_value_decomposition(G_3)
G_3_tilde_U, G_3_tilde_lambda = calculate_singular_value_decomposition(G_3_tilde)

G_4_U, G_4_lambda = calculate_singular_value_decomposition(G_4)
G_4_tilde_U, G_4_tilde_lambda = calculate_singular_value_decomposition(G_4_tilde)

# Neither nor are I ffs
I = calculate_I(P,G_1_U,R_1,R_1_tilde)
I_2 = G_1_lambda + G_1_tilde_lambda


V_1 = np.dot(np.transpose(G_1_U), P).astype(np.float32)
V_2 = np.dot(np.transpose(G_2_U), P).astype(np.float32)
V_3 = np.dot(np.transpose(G_3_U), P).astype(np.float32)
V_4 = np.dot(np.transpose(G_4_U), P).astype(np.float32)

print(V_1[:8].shape)
t = np.dot(V_1[:8], data[class_idxs[0][0]])
print(t.shape, data[class_idxs[0][0]].shape)
