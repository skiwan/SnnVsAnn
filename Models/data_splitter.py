import numpy as np


class DataSplitter():
    def __init__(self, dataset_path, labels_path, target_path, ratio):
        self.dataset = np.load(dataset_path)
        self.labels = np.load(labels_path)
        self.ratio = ratio
        self.target_path = target_path


    def split(self):
        idxs = [i for i in range(len(self.labels))]
        idxs = np.random.permutation(idxs)
        cutoff = int(len(self.labels) * self.ratio)
        train_idxs = idxs[:cutoff]
        validate_idxs = idxs[cutoff:]

        train_data = self.dataset[train_idxs]
        train_labels =  self.labels[train_idxs]

        validate_data = self.dataset[validate_idxs]
        validate_labels = self.labels[validate_idxs]

        np.save(f'{self.target_path}_train_data.npy', train_data)
        np.save(f'{self.target_path}_train_labels.npy', train_labels)
        np.save(f'{self.target_path}_validate_data.npy', validate_data)
        np.save(f'{self.target_path}_validate_labels.npy', validate_labels)