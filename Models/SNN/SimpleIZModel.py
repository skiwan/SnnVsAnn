from IZNeuron import IZNeuron
import numpy as np

class SimpleIZModel(object):
	def __init__(self,*, input_layer_neurons, output_layer_neurons, input_layer_weight_matrix, intermediate_weight_matrix, current_value):
		self.input_layer = [IZNeuron() for i in range(input_layer_neurons)]
		self.input_neuron_count = len(self.input_layer)
		self.input_weight_matrix = input_layer_weight_matrix

		self.intermediate_weight_matrix = intermediate_weight_matrix
		self.output_layer = [IZNeuron() for i in range(output_layer_neurons)]
		self.output_neuron_count = len(self.output_layer)

		self.current_value = current_value

	def single_forward(self, input_sample, timestep=1):

		# multiply input by weight matrix,
		weighted_input = [input_sample*self.input_weight_matrix[n] for n in range(self.input_neuron_count)]
		# sum up to input current
		input_current = [weighted_input[n].sum() for n in range(self.input_neuron_count)]
		# integrate input neurons
		spikes = [self.input_layer[n].integrate(input_current[n], timestep) for n in range(self.input_neuron_count)]
		# if spiked forward, multiply with intermediate weights and forward to output neuron
		spikes = [1 if s is True else 0 for s in spikes]
		intermediate_weighted_current = np.array([spikes * self.intermediate_weight_matrix[n] for n in range(self.output_neuron_count)])
		intermediate_current = [intermediate_weighted_current[n].sum() * self.current_value for n in range(self.output_neuron_count)]
		# return output neuron spike
		return [self.output_layer[n].integrate(intermediate_current[n], timestep) for n in range(self.output_neuron_count)]
