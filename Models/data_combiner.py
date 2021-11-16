import numpy as np

class DataCombiner():
    def __init__(self, dataset_paths, labels_path, target_path):
        self.dataset_paths = dataset_paths
        self.labels = np.load(labels_path)
        self.target_path = target_path

    def combine(self):
        datasets = []
        for path in self.dataset_paths:
            datasets.append(np.load(path))
        whole_set = np.concatenate(datasets, axis=0)
        np.save(f'{self.target_path}_whole_set.npy', whole_set)
        np.save(f'{self.target_path}_whole_labels.npy', self.labels)
