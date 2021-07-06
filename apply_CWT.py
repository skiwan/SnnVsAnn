import numpy as np
import os
import pywt

def average_signal(data, stepsize):
	y = data.shape[0]
	x = data.shape[1]
	if x % stepsize != 0:
		print(f'stepsize {stepsize} not appropriate for input data lenght')
		return data
	prepared = data.reshape((y,int(x/stepsize),stepsize))
	return np.mean(prepared, axis=-1)

def apply_cwt(raw_data, scales, scales2):
	cwt_trials = []
	for t in range(raw_data.shape[0]):
		trial = raw_data[t]
		cwt_trial = []
		for c in range(trial.shape[0]):
			channel_data = trial[c]
			coefs, freq = pywt.cwt(channel_data, scales, 'morl')
			coefs2, freq = pywt.cwt(channel_data, scales2, 'morl')

			coefs = average_signal(coefs, 5)
			coefs2 = average_signal(coefs2, 5)
			combined_cwt = np.concatenate((coefs, coefs2), axis=0)
			cwt_trial.append(combined_cwt)
		cwt_trials.append(cwt_trial)
	cwt_trials = np.asarray(cwt_trials)
	return cwt_trials


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



current_wd = os.getcwd()

configs = [
	[7,15,0.5],
	[16,30,0.5]
]

dt = 0.025

for f_e in files:
	scales = np.arange(*configs[0])
	scales2 = np.arange(*configs[1])
	file_root = f_e[0]
	ev_file_root = f_e[1]

	base_path = os.path.join(current_wd, os.path.join('Raw_Preprocessed', f'{file_root}_car_7_30.npy'))
	label_file_path = os.path.join('Raw_Preprocessed',f'{file_root}_car_labels.txt')
	save_file_base = os.path.join('Raw_Preprocessed_CWT', f'{file_root}_car_7_30.npy')

	ev_base_path = os.path.join(current_wd, os.path.join('Raw_Preprocessed', f'{ev_file_root}_car_7_30.npy'))
	ev_label_file_path = os.path.join('Raw_Preprocessed', f'{ev_file_root}_car_labels.txt')
	ev_save_file_base = os.path.join('Raw_Preprocessed_CWT', f'{ev_file_root}_car_7_30.npy')

	print(f'Applying CWT for {base_path}')


	# load file
	data = np.load(base_path)
	ev_data = np.load(ev_base_path)

	# apply cwt (split and rejoin channels)
	data_cwt = apply_cwt(data, scales, scales2)
	ev_data_cwt = apply_cwt(ev_data, scales, scales2)


	# save file
	np.save(f'{save_file_base}', data_cwt)
	np.save(f'{ev_save_file_base}', ev_data_cwt)

