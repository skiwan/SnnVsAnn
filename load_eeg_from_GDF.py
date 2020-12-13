import mne
import pandas as pd
import numpy as np
from bandpass_filter import butter_bandpass_filter
import os, sys


file_path = 'A02T.gdf'
current_wd = os.getcwd()
base_path = os.path.join(current_wd, 'Preprocessed\BCI4_2a_A02T_car')
frequency = 250
low_pass = 7
high_pass = 30
save_path = f'{base_path}_{low_pass}_{high_pass}.npy'


def get_single_trial(trigger, all_lines, frequency):
	current_trial = []
	for i in range(trigger, trigger+int(frequency)):
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


raw_gdf = mne.io.read_raw_gdf(file_path)
gdf_df = raw_gdf.to_data_frame()
gdf_df = gdf_df.sort_values('time',ascending=True)
gdf_list = gdf_df.drop(labels=['time'], axis=1).values.tolist()

print(gdf_df.head(1))
print(gdf_list[0])

events, events_ids = mne.events_from_annotations(raw_gdf)

"""
768 (6) start of trial -> important
769 - 772 (7-10) onset for cue (class 1-4)
32766 (5) start of a new run
250 hz (*6)
6s trial 
"""

trial_starts = []
trial_labels = []

for sub in events:
	sub_id = sub[2]
	if sub_id == 6:
		trial_starts.append(sub[0])
	if sub_id > 6 and sub_id <= 10:
		trial_labels.append(sub_id)

single_trials = []
for start_idx in trial_starts:
	trial = get_single_trial(start_idx, gdf_list, frequency)
	single_trials.append(trial)

eeg_raw = np.asarray(single_trials)
eeg_filtered = butter_bandpass_filter(eeg_raw, low_pass, high_pass, frequency)
car_eeg = get_CAR_eeg(eeg_filtered)
save_eeg_to_npy(car_eeg, save_path)
with open (f'{base_path}_labels.txt', 'w+') as label_file:
	for label in trial_labels:
		true_label = label - 6
		label_file.write(f'{true_label}\n')