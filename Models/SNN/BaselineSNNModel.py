from SimpleIZModel import SimpleIZModel

class BaselineSNNModel(object):

	def __init__(self,*, class_amount, SIZMParams):
		self.class_amount = class_amount
		self.SIZMs = [SimpleIZModel(**SIZMParams) for i in range(self.class_amount)]

	def classify_trial(self,trial, timestep):
		spikes_c = []
		for model in self.SIZMs:
			model_spikes = [model.single_forward(sample, timestep) for sample in trial]
			spikes_c.append(len([spike for spike in model_spikes if spike is True]))
		return spikes_c.index(max(spikes_c))