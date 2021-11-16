import numpy as np
from math import floor

class DataSplitter():
    def __init__(self, dataset_path, labels_path, target_path, ratio):
        self.dataset = np.load(dataset_path)
        self.labels = np.load(labels_path)
        self.ratio = ratio
        self.target_path = target_path

    # just randomly split
    def __split_random(self):
        idxs = [i for i in range(len(self.labels))]
        idxs = np.random.permutation(idxs)
        cutoff = int(len(self.labels) * self.ratio)
        train_idxs = idxs[:cutoff]
        validate_idxs = idxs[cutoff:]

        train_data = self.dataset[train_idxs]
        train_labels = self.labels[train_idxs]

        validate_data = self.dataset[validate_idxs]
        validate_labels = self.labels[validate_idxs]

        np.save(f'{self.target_path}_train_data.npy', train_data)
        np.save(f'{self.target_path}_train_labels.npy', train_labels)
        np.save(f'{self.target_path}_validate_data.npy', validate_data)
        np.save(f'{self.target_path}_validate_labels.npy', validate_labels)

    # make sure that for train and val, ratio of classes stays the same
    def __split_balanced(self):
        # TODO
        pass

    # makes sure that there 50/50 in train and val per class via copy of smaller class samples
    def __split_balanced_copy(self):
        # get label indexes of class 1
        class_1_idx = [i for i in range(len(self.labels)) if self.labels[i] == 1]
        class_1_idx = np.random.permutation(class_1_idx)
        # get label indexes of class 0
        class_0_idx = [i for i in range(len(self.labels)) if i not in class_1_idx]
        class_0_idx = np.random.permutation(class_0_idx)

        # split them by ratio
        cutoff_1 = floor(len(class_1_idx) * self.ratio)
        cutoff_0 = floor(len(class_0_idx) * self.ratio)

        train_1_idxs = class_1_idx[:cutoff_1]
        validate_1_idxs = class_1_idx[cutoff_1:]

        train_0_idxs = class_0_idx[:cutoff_0]
        validate_0_idxs = class_0_idx[cutoff_0:]
        # replicate class 1 * other classes - 1
        class_ratio = int(len(class_0_idx)/len(class_1_idx))-1
        t_1 = np.asarray(list(train_1_idxs))
        v_1 = np.asarray(list(validate_1_idxs))
        for i in range(class_ratio):
            train_1_idxs = np.concatenate((train_1_idxs,t_1), axis=0)
            validate_1_idxs = np.concatenate((validate_1_idxs,v_1), axis=0)

        # add them together
        train_idx = np.concatenate((train_1_idxs, train_0_idxs))
        val_idx = np.concatenate((validate_1_idxs, validate_0_idxs))

        train_data = self.dataset[train_idx]
        train_labels = self.labels[train_idx]

        validate_data = self.dataset[val_idx]
        validate_labels = self.labels[val_idx]

        np.save(f'{self.target_path}_train_data.npy', train_data)
        np.save(f'{self.target_path}_train_labels.npy', train_labels)
        np.save(f'{self.target_path}_validate_data.npy', validate_data)
        np.save(f'{self.target_path}_validate_labels.npy', validate_labels)


    def split(self, strategy='random'):
        if strategy == 'random':
            self.__split_random()
        if strategy == 'balanced':
            self.__split_balanced()
        if strategy == 'balanced-copy':
            self.__split_balanced_copy()
