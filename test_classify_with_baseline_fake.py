import numpy as np

from Models.SNN.BaselineSNNModel import BaselineSNNModel




def prepare_data(all_data):
	return [d.reshape((d.shape[0],d.shape[2],d.shape[1])) for d in all_data]

if __name__ == "__main__":

	amplifier = 1
	data = [np.ones((360,i,250)) for i in range(1,5)]	
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
	baseline_model = BaselineSNNModel(
		class_amount=class_count, SIZMParams=sm_params)
	data_reshaped = prepare_data(data)	
	
	for i in range(trials):
		full_trial = [data_reshaped[c][i]*amplifier for c in range(class_count)]
		print(baseline_model.classify_trial(full_trial,timestep=1))