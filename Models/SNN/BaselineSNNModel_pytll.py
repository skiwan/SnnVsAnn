import pytorch_lightning as pl
import torch

class TorchBaselineSNNModel(pl.LightningModule):

	def __init__(self, *, class_amount, SIZMParams):
		super().__init__()
		self.class_amount = class_amount
		self.SIZMs = [SimpleIZModel(**SIZMParams[i]) for i in range(self.class_amount)]

	def forward(self, trial, timestep=1):
		spikes_c = []
		for model_n in range(len(self.SIZMs)):
			model = self.SIZMs[model_n]
			trial = full_trial[model_n]
			model_spikes = [model.single_forward(sample, timestep) for sample in trial]
			spikes_c.append(sum([sum(spikes) for spikes in model_spikes]))
		#print(spikes_c)
		return spikes_c.index(max(spikes_c))

	def training_step(self, x, y):
		return 0
		# TODO implement propper training loss

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer
