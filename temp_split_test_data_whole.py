from Models.data_splitter import DataSplitter

dataset_path = '/home/merlinsewina/MaWork/SnnVsAnn/prep/A01T_Prep_whole_set.npy'
labels_path = '/home/merlinsewina/MaWork/SnnVsAnn/prep/A01T_Prep_whole_labels.npy'
target_path = '/home/merlinsewina/MaWork/SnnVsAnn/prep/A01T_Prep_whole'
ratio = 0.7

data_splitter = DataSplitter(dataset_path, labels_path, target_path, ratio)
data_splitter.split('balanced')