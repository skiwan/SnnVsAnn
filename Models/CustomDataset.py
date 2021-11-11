import numpy as np
from torch.utils.data import Dataset

class Dataset(Dataset):

    def __init__(self, label_file, data_file):
        self.data_labels = np.load(label_file)
        self.data_ = np.load(data_file)

    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, idx):
        return self.data[idx], self.data_labels[idx]
