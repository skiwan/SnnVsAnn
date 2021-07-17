import mne
import pandas as pd
import numpy as np
from bandpass_filter import butter_bandpass_filter
import os, sys

"""
768 (6) start of trial -> important
769 - 772 (7-10) onset for cue (class 1-4)
32766 (5) start of a new run
250 hz (*6)
6s trial 
"""

def get_single_trial(trigger, all_lines, frequency, trial_duration):
	current_trial = []
	for i in range(trigger, trigger+int(frequency*trial_duration)):
		current_trial.append(all_lines[i])
	current_trial = np.asarray(current_trial)
	return current_trial

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

def save_eeg_to_npy(eeg_data, file_path_name):
	#with open (file_path_name, 'w+') as eeg_save_file:
	np.save(file_path_name, eeg_data)

def extract_labels_and_trial_starts(events):
	trial_starts = []
	trial_labels = []

	for sub in events:
		sub_id = sub[2]
		if sub_id == 6:
			trial_starts.append(sub[0])
		if sub_id > 6 and sub_id <= 10:
			trial_labels.append(sub_id)
	return trial_starts, trial_labels

def extract_single_trials(trial_starts, gdf_list, frequency, trial_duration):
	single_trials = []
	for start_idx in trial_starts:
		trial = get_single_trial(start_idx, gdf_list, frequency, trial_duration)
		single_trials.append(trial)
	return single_trials

def save_labels_to_file(base_path, trial_labels):
	trial_labels = [t-6 for t in trial_labels]
	np.save(base_path, trial_labels)




"""
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
]

files = [
	[os.path.join(current_wd, 'Datasets\BCICompetitionIV\Data\BCICIV_2a_gdf\A01T.gdf'),'A01T']
	,[os.path.join(current_wd, 'Datasets\BCICompetitionIV\Data\BCICIV_2a_gdf\A02T.gdf'),'A02T']
	,[os.path.join(current_wd, 'Datasets\BCICompetitionIV\Data\BCICIV_2a_gdf\A03T.gdf'),'A03T']
	,[os.path.join(current_wd, 'Datasets\BCICompetitionIV\Data\BCICIV_2a_gdf\A04T.gdf'),'A04T']
	,[os.path.join(current_wd, 'Datasets\BCICompetitionIV\Data\BCICIV_2a_gdf\A05T.gdf'),'A05T']
	,[os.path.join(current_wd, 'Datasets\BCICompetitionIV\Data\BCICIV_2a_gdf\A06T.gdf'),'A06T']
	,[os.path.join(current_wd, 'Datasets\BCICompetitionIV\Data\BCICIV_2a_gdf\A07T.gdf'),'A07T']
	,[os.path.join(current_wd, 'Datasets\BCICompetitionIV\Data\BCICIV_2a_gdf\A08T.gdf'),'A08T']
	,[os.path.join(current_wd, 'Datasets\BCICompetitionIV\Data\BCICIV_2a_gdf\A09T.gdf'),'A09T']
	,[os.path.join(current_wd, 'Datasets\BCICompetitionIV\Data\BCICIV_2a_gdf\A01E.gdf'),'A01E']
	,[os.path.join(current_wd, 'Datasets\BCICompetitionIV\Data\BCICIV_2a_gdf\A02E.gdf'),'A02E']
	,[os.path.join(current_wd, 'Datasets\BCICompetitionIV\Data\BCICIV_2a_gdf\A03E.gdf'),'A03E']
	,[os.path.join(current_wd, 'Datasets\BCICompetitionIV\Data\BCICIV_2a_gdf\A04E.gdf'),'A04E']
	,[os.path.join(current_wd, 'Datasets\BCICompetitionIV\Data\BCICIV_2a_gdf\A05E.gdf'),'A05E']
	,[os.path.join(current_wd, 'Datasets\BCICompetitionIV\Data\BCICIV_2a_gdf\A06E.gdf'),'A06E']
	,[os.path.join(current_wd, 'Datasets\BCICompetitionIV\Data\BCICIV_2a_gdf\A07E.gdf'),'A07E']
	,[os.path.join(current_wd, 'Datasets\BCICompetitionIV\Data\BCICIV_2a_gdf\A08E.gdf'),'A08E']
	,[os.path.join(current_wd, 'Datasets\BCICompetitionIV\Data\BCICIV_2a_gdf\A09E.gdf'),'A09E']
	
]"""

def load_eeg_from_gdf(input_file, base_save_file, low_pass, high_pass, frequency=250, trial_duration=6):
	save_path = f'{base_save_file}_{low_pass}_{high_pass}.npy'
	raw_gdf = mne.io.read_raw_gdf(input_file)
	gdf_df = raw_gdf.to_data_frame()
	gdf_df = gdf_df.sort_values('time',ascending=True)
	gdf_list = gdf_df.drop(labels=['time'], axis=1).values.tolist()

	events, events_ids = mne.events_from_annotations(raw_gdf)
	trial_starts, trial_labels = extract_labels_and_trial_starts(events)
	single_trials = extract_single_trials(trial_starts, gdf_list, frequency, trial_duration)

	eeg_raw = np.asarray(single_trials)
	eeg_raw = np.swapaxes(eeg_raw, 1, 2)
	eeg_filtered = butter_bandpass_filter(eeg_raw, low_pass, high_pass, frequency)
	car_eeg = get_CAR_eeg(eeg_filtered)
	car_eeg = np.nan_to_num(car_eeg, neginf=0, posinf=0)
	for trial in range(car_eeg.shape[0]):
		trial_max = (np.max(np.absolute(car_eeg[trial])))
		car_eeg[trial] = car_eeg[trial] / trial_max
	save_eeg_to_npy(car_eeg, save_path)
	save_labels_to_file(base_save_file, trial_labels)

def main(low_pass, high_pass, input_file, save_file, frequency=250, trial_duration=6):
	load_eeg_from_gdf(low_pass, high_pass, input_file, save_file, frequency, trial_duration)

if __name__ == "__main__":
	main(*sys.argv[1:])