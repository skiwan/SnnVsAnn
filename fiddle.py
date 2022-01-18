import os
import shutil
from Scripts.Utils import create_temp_folder, delete_temp_folder, create_folder_if_not_exists, delete_folder
from load_eeg_from_GDF import load_eeg_from_gdf
from apply_CSP import apply_csp
from normalize_feature_extraction import apply_normlized_feature_extraction
from Models.data_splitter import DataSplitter


file_directory = os.path.dirname(os.path.abspath(__file__))
file_directory = os.path.join(file_directory, 'Scripts/')
base_save_path = os.path.join(file_directory, 'temp/')

n='01'
experiment_name = f'Binary_SNN_A{n}'
experiment_description = 'One vs Rest classification of GDF based motor imagery classifcation on SNN Architecture. ' \
                         'Applies Error averaging, CSP and Normalization'
train_file_name = f'A{n}T.gdf'
eval_file_name = f'A{n}E.gdf'
eval_label_file_name = f'A{n}E_labels.npy'
device = f'cuda'

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
batch_size = 16
shuffle = True
workers = 1
max_epochs = 5
model_channels = 8
model_classes = 2

model_learning_rate = 0.01
model_weight_decay = 0.1
data_cut_front = 0
data_cut_back = 200
experiment_name = f'{experiment_name}_learnrate{model_learning_rate}_weightdec{model_weight_decay}_cutoff{data_cut_front}_{data_cut_back}'
save_model = True
destination_path = os.path.join(file_directory, 'Experiments')
destination_path = os.path.join(destination_path, f'{experiment_name}')
create_folder_if_not_exists(destination_path)
destination_path = f'{destination_path}/'

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
i = 1
# apply normalize without extraction in temp folder
apply_normlized_feature_extraction(f'{base_save_path}csp_train_class{i}.npy', f'{base_save_path}raw_train_labels.npy',
                                   f'{base_save_path}normalized_train',
                                   f'{base_save_path}csp_eval_class{i}.npy', f'{base_save_path}raw_eval_labels.npy',
                                   f'{base_save_path}normalized_eval',
                                   i, extract=extract_features)

# Prepare Train and Val sets
data_splitter = DataSplitter(f'{base_save_path}cwt_train_class{i}.npy', f'{base_save_path}normalized_train_class{i}_labels.npy', f'{base_save_path}_class{i}', split_ratio)
data_splitter.split(splitting_strategy)