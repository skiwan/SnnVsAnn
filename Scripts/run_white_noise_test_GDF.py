import os, shutil
from Utils import create_temp_folder
from load_eeg_from_GDF import load_eeg_from_gdf
from apply_CSP import apply_csp
from normalize_feature_extraction import apply_normlized_feature_extraction
from apply_CWT import apply_cwt
from Models.data_splitter import DataSplitter
from noise_configs import white_noise_values

from model_configs import base_model_config
# Prep

file_directory = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(file_directory)
base_save_path = os.path.join(file_directory, 'temp/')

# Create temp folder if not exists
create_temp_folder(file_directory)

experiment_description = f"White Noise Test Data Set: {1} noise params: {2}"

experiment_description = experiment_description
train_file_name = train_file_name
eval_file_name = eval_file_name
eval_label_file_name = eval_label_file_name

base_file_path = os.path.join(parent_folder, 'original_data/Datasets/BCICompetitionIV/Data/BCICIV_2a_gdf/')
base_label_path = os.path.join(parent_folder, 'savecopywithlabels/')

raw_train_file_name = os.path.join(base_file_path, train_file_name)
raw_eval_file_name = os.path.join(base_file_path, eval_file_name)


####
low_pass = base_model_config["low_pass"]
high_pass = base_model_config["high_pass"]
extract_features = base_model_config["extract_features"]
scale_1 = base_model_config["scale_1"]
scale_2 = base_model_config["scale_2"]
split_ratio = base_model_config["split_ratio"]
splitting_strategy = base_model_config["split_ratio"]

####

# Generate normal Data
# load raw file into temp folder
load_eeg_from_gdf(low_pass, high_pass, raw_train_file_name, f'{base_save_path}raw_train', frequency=250,
                  trial_duration=6)
load_eeg_from_gdf(low_pass, high_pass, raw_eval_file_name, f'{base_save_path}raw_eval', frequency=250,
                  trial_duration=6)

# move and prepare labels into temp folder
shutil.copyfile(f'{base_label_path}{eval_label_file_name}', f'{base_save_path}raw_eval_labels.npy')

# apply CPS in temp folder
apply_csp(f'{base_save_path}raw_train_{low_pass}_{high_pass}.npy', f'{base_save_path}raw_train_labels.npy',
          f'{base_save_path}raw_eval_{low_pass}_{high_pass}.npy', f'{base_save_path}raw_eval_labels.npy',
          f'{base_save_path}csp_train', f'{base_save_path}csp_eval',
          low_pass, high_pass)

for i in range(1, 5):
    # apply normalize without extraction in temp folder
    apply_normlized_feature_extraction(f'{base_save_path}csp_train_class{i}.npy',
                                       f'{base_save_path}raw_train_labels.npy',
                                       f'{base_save_path}normalized_train',
                                       f'{base_save_path}csp_eval_class{i}.npy', f'{base_save_path}raw_eval_labels.npy',
                                       f'{base_save_path}normalized_eval',
                                       i, extract=extract_features)

    # apply CWT
    apply_cwt(f'{base_save_path}normalized_train_class{i}.npy', f'{base_save_path}cwt_train_class{i}.npy',
              scale_1[0], scale_1[1], scale_1[2], scale_2[0], scale_2[1], scale_2[2])
    apply_cwt(f'{base_save_path}normalized_eval_class{i}.npy', f'{base_save_path}cwt_eval_class{i}.npy',
              scale_1[0], scale_1[1], scale_1[2], scale_2[0], scale_2[1], scale_2[2])

    # Prepare Train and Val sets SNN
    data_splitter = DataSplitter(f'{base_save_path}normalized_train_class{i}.npy',
                                 f'{base_save_path}normalized_train_class{i}_labels.npy', f'{base_save_path}_snn_class{i}',
                                 split_ratio)
    data_splitter.split(splitting_strategy)

    # Prepare Train and Val sets ANN
    data_splitter = DataSplitter(f'{base_save_path}cwt_train_class{i}.npy',
                                 f'{base_save_path}normalized_train_class{i}_labels.npy', f'{base_save_path}_ann_class{i}',
                                 split_ratio)
    data_splitter.split(splitting_strategy)

# Train SNN on normal data
# Evaluate and save baseline


# Train ANN on normal data
# Evaluate and save baseline


    # Generate noisey Data
    # Evaluate and save data

# Save all stats
# Generate Graph for ANN and noise to eval
# Generate Graph for SNN and noise to eval
