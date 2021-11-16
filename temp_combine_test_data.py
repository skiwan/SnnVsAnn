from Models.data_combiner import DataCombiner

dataset_paths = [
    '/home/merlinsewina/MaWork/SnnVsAnn/prep/A01T_7_30_class1_CWT.npy'
    ,'/home/merlinsewina/MaWork/SnnVsAnn/prep/A01T_7_30_class2_CWT.npy'
    ,'/home/merlinsewina/MaWork/SnnVsAnn/prep/A01T_7_30_class3_CWT.npy'
    ,'/home/merlinsewina/MaWork/SnnVsAnn/prep/A01T_7_30_class4_CWT.npy'
]

labels_path = '/home/merlinsewina/MaWork/SnnVsAnn/prep/A01T_raw_labels.npy'
target_path = '/home/merlinsewina/MaWork/SnnVsAnn/prep/A01T_Prep'

data_combiner = DataCombiner(dataset_paths, labels_path, target_path)
data_combiner.combine()