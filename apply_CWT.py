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

def use_cwt(raw_data, scales, scales2):
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
['BCI4_2a_A01T_car_7_30_full','BCI4_2a_A01E_car_7_30_full']
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
"""configs = [
	[7,15,0.5],
	[16,30,0.5]
]"""



def apply_cwt(input_file, save_file, l1, h1, s1, l2, h2, s2, dt=0.025):
	scales = [l1, h1, s1]
	scales2 = [l2, h2, s2]
	print(f'Applying CWT for {input_file}')
	# load file
	data = np.load(input_file)

	# apply cwt (split and rejoin channels)
	data_cwt = use_cwt(data, scales, scales2)

	# save file
	np.save(f'{save_file}', data_cwt)

def main(input_file, save_file, l1, h1, s1, l2, h2, s2, dt=0.025):
	apply_cwt(input_file, save_file, l1, h1, s1, l2, h2, s2, dt)

if __name__ == "__main__":
	main(*sys.argv[1:])
