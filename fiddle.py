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
    # Prepare Train and Val sets
    data_splitter = DataSplitter(f'{base_save_path}normalized_train_class{i}.npy',
                                 f'{base_save_path}normalized_train_class{i}_labels.npy', f'{base_save_path}_class{i}',
                                 split_ratio)
    data_splitter.split(splitting_strategy)

# Load all 4 best models
# Load eval file set
# for each class pick 4 samples
# for each model, run all 4*4 samples
    # Save spike trains
    # Save spike trains as txt and as pngs via matplotlib

models = []
save_model = True

for i in range(1,5):
    model_name = f'{base_save_path}{experiment_name}_class{i}_model'

    statistics, e_loss, eval_acc, eval_kappa, best_val_epoch = run_binary_classification(
        batch_size, shuffle, workers, 300,
        f'{base_save_path}_class{i}_train_data.npy', f'{base_save_path}_class{i}_train_labels.npy',
        f'{base_save_path}_class{i}_validate_data.npy', f'{base_save_path}_class{i}_validate_labels.npy',
        f'{base_save_path}normalized_eval_class{i}.npy', f'{base_save_path}normalized_eval_class{i}_labels.npy',
        model_channels, model_classes,
        model_learning_rate, model_weight_decay, save_model,
        f'{base_save_path}{experiment_name}_class{i}_model', device
    )

    # Save Training Run Statistics
    train_statistics = {
        'epoch': statistics[0]
        , 'train_loss': statistics[1]
        , 'train_acc': statistics[2]
        , 'val_loss': statistics[3]
        , 'val_acc': statistics[4]
    }

    # load saved best val model and apply eval set
    if save_model:
        e_loss, eval_acc, eval_kappa = load_and_run_eval(
            f'{base_save_path}{experiment_name}_class{i}_model.pth',
            f'{base_save_path}_class{i}_train_data.npy', f'{base_save_path}_class{i}_train_labels.npy'
            , f'{base_save_path}normalized_eval_class{i}.npy', f'{base_save_path}normalized_eval_class{i}_labels.npy'
            , model_channels, model_classes, device)

best_acc, best_kappa, last_acc, last_kappa, model_output = multiclass_run_data(base_save_path, experiment_name, 4
                                                                    , model_channels,
                                                                    model_classes, 'cpu')

print(f'Multiclass Accuarcy best: {best_acc}, last: {last_acc}')
overall_best_acc = max(best_acc, last_acc)
print(overall_best_acc)

activity_spikes_best = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
activity_spikes_last = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]

# For each model getlast and best
for c in range(4):
    best_c = model_output[c][0]
    last_c = model_output[c][1]

    # for each class
    for l in range(4):
        samples = best_c[l][:4]
        activity_spikes_best[l][c] = samples
        samples = last_c[l][:4]
        activity_spikes_last[c][l] = samples

for l in range(4):
    all_best_trains = np.asarray(activity_spikes_best[l])
    np.save(f'_best_models_label_{l}_activity.npy', all_best_trains)
    all_last_trains = np.asarray(activity_spikes_last[l])
    np.save(f'_last_models_label_{l}_activity.npy', all_last_trains)

