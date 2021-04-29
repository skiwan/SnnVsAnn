from .SimpleIZModel import SimpleIZModel

class BaselineSNNModel(object):

	def __init__(self,*, class_amount, SIZMParams):
		self.class_amount = class_amount
		self.SIZMs = [SimpleIZModel(**SIZMParams[i]) for i in range(self.class_amount)]

	def classify_trial(self,full_trial, timestep):
		spikes_c = []
		for model_n in range(len(self.SIZMs)):
			model = self.SIZMs[model_n]
			trial = full_trial[model_n]
			model_spikes = [model.single_forward(sample, timestep) for sample in trial]
			spikes_c.append(sum([sum(spikes) for spikes in model_spikes]))
		print(spikes_c)
		return spikes_c.index(max(spikes_c))