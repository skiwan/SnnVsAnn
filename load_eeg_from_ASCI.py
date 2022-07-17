import os, sys
import numpy as np
from bandpass_filter import butter_bandpass_filter

def convert_to_float_row(line):
	row = []
	for number in line.split(' '):
		if len(number) > 0:
			row.append(float(number))
	return row

def get_trigger_indexes(trigger_file_path):
	trigger_idxs = []
	with open(trigger_file_path, 'r') as eeg_trial_starts:
		lines = eeg_trial_starts.readlines()
		for l in lines:
			trigger_idxs.append(int(l))
	return trigger_idxs

def get_single_trial(trigger, all_lines, frequency, trial_duration):
	current_trial = []
	for i in range(trigger, trigger+int(frequency*trial_duration)):
		current_trial.append(convert_to_float_row(all_lines[i]))
	current_trial = np.asarray(current_trial)
	return current_trial

def load_asci_eeg_to_np_array(raw_file_path, trigger_file_path, frequency, trial_duration):
	single_trials = []
	trigger_idxs = get_trigger_indexes(trigger_file_path)
	with open(raw_file_path, 'r') as eeg_data:
		lines = eeg_data.readlines()
	
	for trigger in trigger_idxs:
		trial = get_single_trial(trigger, lines, frequency, trial_duration)
		single_trials.append(trial)

	return np.asarray(single_trials)


def get_filtered_eeg(raw_file_path, trigger_file_path, low_pass, high_pass, frequency, trial_duration, noise_fn, noise_fn_params):
	eeg_raw = load_asci_eeg_to_np_array(raw_file_path, trigger_file_path, frequency, trial_duration)
	eeg_raw = np.swapaxes(eeg_raw, 1, 2)
	if noise_fn and noise_fn_params:
		eeg_raw = noise_fn(eeg_raw, **noise_fn_params)
	eeg_filtered = butter_bandpass_filter(eeg_raw, low_pass, high_pass, frequency)
	return eeg_filtered

def get_CAR_eeg(filtered_eeg):
	total_sum = 0
	total_count = 0
	for trial in filtered_eeg:
		for row in trial:
			total_sum += np.nansum(row)
			total_count += len(row)
	average = total_sum / total_count
	car_eeg = filtered_eeg - average
	return car_eeg

def get_true_labels(true_label_path):
	labels = []
	with open(true_label_path, 'r') as true_labels:
		lines = true_labels.readlines()
	for line in lines:
		labels.append(int(line))
	return labels

def save_eeg_to_npy(eeg_data, file_path_name):
	np.save(file_path_name, eeg_data)

"""low_pass = 7
high_pass = 30
test_file = os.path.join(current_wd, 'Datasets\BCICompetitionIII\Data\BCIIII_DataSetIIIa-Sub_1_k3b_ascii\k3b_s.txt')
trigger_path = os.path.join(current_wd, 'Datasets\BCICompetitionIII\Data\BCIIII_DataSetIIIa-Sub_1_k3b_ascii\k3b_HDR_TRIG.txt')
save_path = os.path.join(current_wd, 'Preprocessed\BCI3a_k3b_car_7_30.npy')

#true_labels_path = os.path.join(current_wd, 'Datasets\BCICompetitionIII\Data\\true_label_k3.txt')
#eeg_labels = get_true_labels(true_labels_path)

eeg_filtered = get_filtered_eeg(test_file,trigger_path, low_pass, high_pass, frequency)
car_eeg = get_CAR_eeg(eeg_filtered)
save_eeg_to_asci(car_eeg, save_path)

files = [
	[os.path.join(current_wd, 'Datasets\BCICompetitionIII\Data\BCIIII_DataSetIIIa-Sub_1_k3b_ascii\k3b_s.txt')
	,os.path.join(current_wd, 'Datasets\BCICompetitionIII\Data\BCIIII_DataSetIIIa-Sub_1_k3b_ascii\k3b_HDR_TRIG.txt')
	,os.path.join(current_wd, 'Preprocessed\BCI3_3a_k3b_car')]
	,[os.path.join(current_wd, 'Datasets\BCICompetitionIII\Data\BCIIII_DataSetIIIa-Sub_2_k6b_ascii\k6b_s.txt')
	,os.path.join(current_wd, 'Datasets\BCICompetitionIII\Data\BCIIII_DataSetIIIa-Sub_2_k6b_ascii\k6b_HDR_TRIG.txt')
	,os.path.join(current_wd, 'Preprocessed\BCI3_3a_k6b_car')]
	,[os.path.join(current_wd, 'Datasets\BCICompetitionIII\Data\BCIIII_DataSetIIIa-Sub_3_l1b_ascii\l1b_s.txt')
	,os.path.join(current_wd, 'Datasets\BCICompetitionIII\Data\BCIIII_DataSetIIIa-Sub_3_l1b_ascii\l1b_HDR_TRIG.txt')
	,os.path.join(current_wd, 'Preprocessed\BCI3_3a_l1b_car')]
]
configs = [
	[7, 30]
	,[7,14]
	,[14,30]
	,[7,12]
	,[12,17]
	,[17,22]
	,[22,27]
	,[27,32]
	,[4,8]
	,[8,12]
	,[12,16]
	,[16,20]
	,[20,24]
	,[24,28]
	,[28,32]
	,[32,36]
	,[36,40]
	,[9,14]
	,[11,16]
	,[13,18]
	,[15,20]
	,[19,24]
	,[21,26]
	,[23,28]
	,[25,30]
]"""

def load_eeg_from_asci(raw_input_file, trigger_input_file, save_file, low_pass, high_pass, frequency=250, trial_duration=7, noise_fn=None, noise_fn_params={}):
	save_path = f'{save_file}_{low_pass}_{high_pass}.npy'
	eeg_filtered = get_filtered_eeg(raw_input_file, trigger_input_file, low_pass, high_pass, frequency, trial_duration, noise_fn, noise_fn_params)
	car_eeg = get_CAR_eeg(eeg_filtered)
	car_eeg = np.nan_to_num(car_eeg, neginf=0, posinf=0)
	for trial in range(car_eeg.shape[0]):
		trial_max = (np.max(np.absolute(car_eeg[trial])))
		car_eeg[trial] = car_eeg[trial] / trial_max
	save_eeg_to_npy(car_eeg, save_path)



def main(raw_input_file, trigger_input_file, save_file, low_pass, high_pass, frequency=250, trial_duration=7):
	load_eeg_from_asci(raw_input_file, trigger_input_file, save_file, int(low_pass), int(high_pass), frequency, trial_duration)

if __name__ == "__main__":
	main(*sys.argv[1:])