from Models.data_splitter import DataSplitter

dataset_path = '/home/merlinsewina/MaWork/SnnVsAnn/prep/A01T_7_30_class1_CWT.npy'
labels_path = '/home/merlinsewina/MaWork/SnnVsAnn/prep/A01T_7_30_Features_class1_labels.npy'
target_path = '/home/merlinsewina/MaWork/SnnVsAnn/prep/A01T_Prep'
ratio = 0.7

data_splitter = DataSplitter(dataset_path, labels_path, target_path, ratio)
data_splitter.split('balanced-copy')