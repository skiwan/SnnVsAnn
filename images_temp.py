import os
import shutil
from Scripts.Utils import create_temp_folder, delete_temp_folder, create_folder_if_not_exists, delete_folder
from Scripts.binary_class_snn_run import run_binary_classification, load_and_run_eval
from load_eeg_from_GDF import load_eeg_from_gdf
from apply_CSP import apply_csp
from normalize_feature_extraction import apply_normlized_feature_extraction
from Models.data_splitter import DataSplitter
from Scripts.multi_class_snn_run import main as multiclass_run
from Scripts.multi_class_snn_run import main_return_data as multiclass_run_data
from apply_CWT import apply_cwt
import matplotlib.pyplot as plt
from matplotlib import image

import numpy as np

file_directory = os.path.dirname(os.path.abspath(__file__))
file_directory = os.path.join(file_directory, 'Scripts/')
base_save_path = os.path.join(file_directory, 'temp/')

n='03'
experiment_name = f'Binary_SNN_A{n}'
experiment_description = 'One vs Rest classification of GDF based motor imagery classifcation on SNN Architecture. ' \
                            'Applies Error averaging, CSP and Normalization'
train_file_name = f'A{n}T.gdf'
eval_file_name = f'A{n}E.gdf'
eval_label_file_name = f'A{n}E_labels.npy'
device = f'cpu'

# Create temp folder if not exists
create_temp_folder(file_directory)
train_file_name = train_file_name
eval_file_name = eval_file_name
eval_label_file_name = eval_label_file_name

base_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'original_data/Datasets/BCICompetitionIV/Data/BCICIV_2a_gdf/')
base_label_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'savecopywithlabels/')

low_pass = 7
high_pass = 30
raw_train_file_name = os.path.join(base_file_path, train_file_name)
raw_eval_file_name = os.path.join(base_file_path, eval_file_name)
extract_features = False
scale_1 = [7,15,0.5]
scale_2 = [16,30,0.5]
split_ratio = 0.7
splitting_strategy = 'balanced-copy'
batch_size = 64
shuffle = True
workers = 1
model_channels = 8
model_classes = 2

model_learning_rate = 0.01
model_weight_decay = 0.0
"""
load_eeg_from_gdf(low_pass, high_pass, raw_train_file_name, f'{base_save_path}raw_train', frequency=250, trial_duration=6)
load_eeg_from_gdf(low_pass, high_pass, raw_eval_file_name, f'{base_save_path}raw_eval', frequency=250,
                      trial_duration=6)

# move and prepare labels into temp folder
shutil.copyfile(f'{base_label_path}{eval_label_file_name}', f'{base_save_path}raw_eval_labels.npy')

# apply CPS in temp folder
apply_csp(f'{base_save_path}raw_train_{low_pass}_{high_pass}.npy', f'{base_save_path}raw_train_labels.npy',
          f'{base_save_path}raw_eval_{low_pass}_{high_pass}.npy', f'{base_save_path}raw_eval_labels.npy',
          f'{base_save_path}csp_train', f'{base_save_path}csp_eval',
          low_pass, high_pass)

# for each of the datasets
# apply normalize without extraction in temp folder
for i in range(1, 5):
    # apply normalize without extraction in temp folder
    apply_normlized_feature_extraction(f'{base_save_path}csp_train_class{i}.npy',
                                       f'{base_save_path}raw_train_labels.npy',
                                       f'{base_save_path}normalized_train',
                                       f'{base_save_path}csp_eval_class{i}.npy', f'{base_save_path}raw_eval_labels.npy',
                                       f'{base_save_path}normalized_eval',
                                       i, extract=extract_features)
    apply_cwt(f'{base_save_path}normalized_train_class{i}.npy', f'{base_save_path}cwt_train_class{i}.npy',
              scale_1[0], scale_1[1], scale_1[2], scale_2[0], scale_2[1], scale_2[2])
    apply_cwt(f'{base_save_path}normalized_eval_class{i}.npy', f'{base_save_path}cwt_eval_class{i}.npy',
              scale_1[0], scale_1[1], scale_1[2], scale_2[0], scale_2[1], scale_2[2])

    # Prepare Train and Val sets
    data_splitter = DataSplitter(f'{base_save_path}cwt_train_class{i}.npy',
                                 f'{base_save_path}normalized_train_class{i}_labels.npy', f'{base_save_path}_class{i}',
                                 split_ratio)
    data_splitter.split(splitting_strategy)"""

# Load CSP EEG DAtA
csp = np.load(f'{base_save_path}normalized_train_class1.npy')
# Select first sample and plot it
s1 = csp[0][0]
plt.plot(s1)
plt.savefig('CSPNormalizedSample.jpg', dpi=400)
# Load CWT EEG Data
cwt = np.load(f'{base_save_path}cwt_train_class1.npy')
s2 = cwt[0][0]
image.imsave("CWTSample.jpg", s2, dpi=400, cmap=image.cm.gray)
# Select first sample and plot it
# Save both
print("a")