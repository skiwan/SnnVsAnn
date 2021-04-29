import numpy as np
import pathlib
import os

from Models.SNN.BaselineSNNModel import BaselineSNNModel


file_path = pathlib.Path(__file__).parent.absolute()
train_data_path = os.path.join(file_path, 'Normalized_Extracted')

train_file = 'BCI3_3a_k3b_car_1_class'


def prepare_data(all_data):
	return [d.reshape((d.shape[0],d.shape[2],d.shape[1])) for d in all_data]

if __name__ == "__main__":
	amplifier = 10
	
	data = [np.load(os.path.join(train_data_path,f'{train_file}{c}.npy')) for c in range(1,5)]
	class_features = [d.shape[1] for d in data]
	trials = data[0].shape[0]
	class_count = len(data)
	sm_params = []
	for i in range(class_count):
		sm_params.append(
			{ 'input_layer_neurons':2, 
		'output_layer_neurons':1, 
		'input_layer_weight_matrix':np.array([[2 for l in range(class_features[i])],[1 for l in range(class_features[i])]]), 
		'intermediate_weight_matrix':np.array([1,1]),
		'amplifier' : amplifier,
		'current_value' : 30}
		)
	#sm_params[-1]['intermediate_weight_matrix']= np.array([5,5])
	baseline_model = BaselineSNNModel(
		class_amount=class_count, SIZMParams=sm_params)
	data_reshaped = prepare_data(data)
	
	for i in range(trials):
		full_trial = [data_reshaped[c][i]*amplifier for c in range(class_count)]
		print(baseline_model.classify_trial(full_trial,timestep=1))