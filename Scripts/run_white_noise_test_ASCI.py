import os, shutil
from Utils import create_temp_folder
from load_eeg_from_ASCI import load_eeg_from_asci
from apply_CSP import apply_csp
from normalize_feature_extraction import apply_normlized_feature_extraction
from apply_CWT import apply_cwt
from Models.data_splitter import DataSplitter
from noise_configs import white_noise_values
from apply_white_noise import apply_white_noise
from Utils import create_folder_if_not_exists
import json
import time
import numpy as np

def get_dataset_idxs(untrue_labels_file_path):
    labels = np.load(untrue_labels_file_path)
    eval_idx = []
    train_idx = []
    for i in range(labels.shape[0]):
        if labels[i] == -1:
            eval_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, eval_idx

def generate_train_and_eval_data(data_path, train_idx, eval_idx, save_path, low_pass, high_pass):
    data = np.load(data_path)
    train_data = data[train_idx]
    eval_data = data[eval_idx]
    np.save(f'{save_path}raw_train_{low_pass}_{high_pass}.npy', train_data)
    np.save(f'{save_path}raw_eval_{low_pass}_{high_pass}.npy', eval_data)

def generate_train_and_eval_labels(label_path, train_idx, eval_idx, save_path):
    labels = np.load(label_path)
    train_labels = labels[train_idx]
    eval_labels = labels[eval_idx]
    np.save(f'{save_path}raw_train_labels.npy', train_labels)
    np.save(f'{save_path}raw_eval_labels.npy', eval_labels)



from training_helper_gdf import train_model_ann, train_model_snn, eval_ann_model, eval_snn_model

from model_configs import base_model_config, asci_models
# Prep

file_directory = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(file_directory)
base_save_path = os.path.join(file_directory, 'temp/')

# Create temp folder if not exists
create_temp_folder(file_directory)

for data_set_name, model_c in asci_models.items():
    start = time.time()

    subject_name = data_set_name.split("_")[0]
    subject_nr = data_set_name.split("_")[1]

    base_file_path = os.path.join(parent_folder, 'original_data/Datasets/BCICompetitionIII/Data/')
    base_label_path = os.path.join(parent_folder, 'savecopywithlabels/')
    subject_folder = os.path.join(base_file_path, f'BCIIII_DataSetIIIa-Sub_{subject_nr}_{subject_name}_ascii')


    train_file_name = os.path.join(subject_folder, f'{subject_name}_s.txt')
    trigger_file = os.path.join(subject_folder, f'{subject_name}_HDR_TRIG.txt')
    untrue_label_file = os.path.join(subject_folder, f'{subject_name}_HDR_Classlabel.txt')
    true_label_file = os.path.join(base_label_path, f'true_labels_{subject_name[:-1]}.npy')

    low_pass = base_model_config["low_pass"]
    high_pass = base_model_config["high_pass"]
    extract_features = base_model_config["extract_features"]
    scale_1 = base_model_config["scale_1"]
    scale_2 = base_model_config["scale_2"]
    split_ratio = base_model_config["split_ratio"]
    splitting_strategy = base_model_config["splitting_strategy"]

    ####

    # Generate normal Data
    # load raw file into temp folder
    load_eeg_from_asci(train_file_name, trigger_file, f'{base_save_path}raw', low_pass, high_pass)

    # move and prepare labels into temp folder
    shutil.copyfile(f'{true_label_file}', f'{base_save_path}true_labels.npy')
    shutil.copyfile(f'{untrue_label_file}', f'{base_save_path}untrue_labels.txt')
    with open(f'{base_save_path}untrue_labels.txt', 'r+') as untrue_labels:
        lines = untrue_labels.readlines()
        lines = [int(l.replace('\n','')) if 'NaN' not in l else -1 for l in lines ]
        np.save(f'{base_save_path}untrue_labels.npy', lines)

        # Seperate data based on untrue labels into train and eval dataset
    train_idx, eval_idx = get_dataset_idxs(f'{base_save_path}untrue_labels.npy')
    generate_train_and_eval_data(f'{base_save_path}raw_{low_pass}_{high_pass}.npy', train_idx, eval_idx,
                                 base_save_path, low_pass, high_pass)
    # Seperate Labels into train and eval labels
    generate_train_and_eval_labels(f'{base_save_path}true_labels.npy', train_idx, eval_idx, base_save_path)

    # apply CPS in temp folder
    apply_csp(f'{base_save_path}raw_train_{low_pass}_{high_pass}.npy', f'{base_save_path}raw_train_labels.npy',
              f'{base_save_path}raw_eval_{low_pass}_{high_pass}.npy', f'{base_save_path}raw_eval_labels.npy',
              f'{base_save_path}csp_train', f'{base_save_path}csp_eval',
              low_pass, high_pass)

    for i in range(1, 5):
        # apply normalize without extraction in temp folder
        apply_normlized_feature_extraction(f'{base_save_path}csp_train_class{i}.npy', f'{base_save_path}raw_train_labels.npy',
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
                                     f'{base_save_path}normalized_train_class{i}_labels.npy', f'{base_save_path}snn_class{i}',
                                     split_ratio)
        data_splitter.split(splitting_strategy)

        # Prepare Train and Val sets ANN
        data_splitter = DataSplitter(f'{base_save_path}cwt_train_class{i}.npy',
                                     f'{base_save_path}normalized_train_class{i}_labels.npy', f'{base_save_path}ann_class{i}',
                                     split_ratio)
        data_splitter.split(splitting_strategy)

    experiment_description = f"White Noise Test Data Set: {data_set_name} base Truth"

    experiment_config_base = {
        "base_save_path": base_save_path,
        "file_directory": file_directory,
        "train_file_name": train_file_name,
        "eval_file_name": train_file_name,
        "experiment_file_name": f"{data_set_name}_basetrain_whitenoise",
        "device": "cpu",
    }

    # Prepare SNN config
    ann_config = {
        "experiment_description": f"{experiment_description} ANN",
        "experiment_name": f"{data_set_name}_basetrain_whitenoise_ann",
    }
    ann_config.update(experiment_config_base)
    ann_config.update(base_model_config)
    ann_config.update(model_c["ANN"])
    # Train ANN on normal data
    ann_pre = train_model_ann(ann_config)



    snn_config = {
        "experiment_description": f"{experiment_description} SNN",
        "experiment_name": f"{data_set_name}_basetrain_whitenoise_snn",
    }
    snn_config.update(experiment_config_base)
    snn_config.update(base_model_config)
    snn_config.update(model_c["SNN"])
    # Train SNN on normal data
    snn_pre = train_model_snn(snn_config)

    print(f"Best ACC after normal Training ann: {ann_pre}")
    print(f"Best ACC after normal Training snn: {snn_pre}")

    destination_path = os.path.join(file_directory, 'Experiments')
    destination_path = os.path.join(
        destination_path, f"{data_set_name}_whitenoise_test"
    )
    create_folder_if_not_exists(destination_path)
    destination_path = f'{destination_path}/'

    ann_eval_config = {'base_save_path': ann_config['base_save_path'], 'experiment_name': ann_config['experiment_name'],
                       'model_channels': ann_config['model_channels'], 'model_classes': ann_config['model_classes'],
                       'model_dropout': ann_config['dropout'], "data_cut_front": model_c["ANN"]["cut_off"][0],
                       "data_cut_back": model_c["ANN"]["cut_off"][1]}

    snn_eval_config = {'base_save_path': snn_config['base_save_path'], 'experiment_name': snn_config['experiment_name'],
                       'model_channels': snn_config['model_channels'], 'model_classes': snn_config['model_classes']}

    for noise_c in white_noise_values:
        # Generate noisey Data
        load_eeg_from_asci(train_file_name, trigger_file, f'{base_save_path}raw', low_pass, high_pass, noise_fn=apply_white_noise, noise_fn_params=noise_c)

        # move and prepare labels into temp folder
        shutil.copyfile(f'{true_label_file}', f'{base_save_path}true_labels.npy')
        shutil.copyfile(f'{untrue_label_file}', f'{base_save_path}untrue_labels.txt')
        with open(f'{base_save_path}untrue_labels.txt', 'r+') as untrue_labels:
            lines = untrue_labels.readlines()
            lines = [int(l.replace('\n', '')) if 'NaN' not in l else -1 for l in lines]
            np.save(f'{base_save_path}untrue_labels.npy', lines)

            # Seperate data based on untrue labels into train and eval dataset
        train_idx, eval_idx = get_dataset_idxs(f'{base_save_path}untrue_labels.npy')
        generate_train_and_eval_data(f'{base_save_path}raw_{low_pass}_{high_pass}.npy', train_idx, eval_idx,
                                     base_save_path, low_pass, high_pass)
        # Seperate Labels into train and eval labels
        generate_train_and_eval_labels(f'{base_save_path}true_labels.npy', train_idx, eval_idx, base_save_path)

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
                                               f'{base_save_path}csp_eval_class{i}.npy',
                                               f'{base_save_path}raw_eval_labels.npy',
                                               f'{base_save_path}normalized_eval',
                                               i, extract=extract_features)

            # apply CWT
            apply_cwt(f'{base_save_path}normalized_train_class{i}.npy', f'{base_save_path}cwt_train_class{i}.npy',
                      scale_1[0], scale_1[1], scale_1[2], scale_2[0], scale_2[1], scale_2[2])
            apply_cwt(f'{base_save_path}normalized_eval_class{i}.npy', f'{base_save_path}cwt_eval_class{i}.npy',
                      scale_1[0], scale_1[1], scale_1[2], scale_2[0], scale_2[1], scale_2[2])

            # Prepare Train and Val sets SNN
            data_splitter = DataSplitter(f'{base_save_path}normalized_train_class{i}.npy',
                                         f'{base_save_path}normalized_train_class{i}_labels.npy',
                                         f'{base_save_path}snn_class{i}',
                                         split_ratio)
            data_splitter.split(splitting_strategy)

            # Prepare Train and Val sets ANN
            data_splitter = DataSplitter(f'{base_save_path}cwt_train_class{i}.npy',
                                         f'{base_save_path}normalized_train_class{i}_labels.npy',
                                         f'{base_save_path}ann_class{i}',
                                         split_ratio)
            data_splitter.split(splitting_strategy)
            # Evaluate and save data



        best_acc, best_kappa, last_acc, last_kappa = eval_ann_model(
            **ann_eval_config
        )

        white_noise_strength = noise_c["noise_strength_percent"]

        ann_test_results = {
            "ModelType": "ANN",
            "Best_ACC": best_acc,
            "Best_Kappa": best_kappa,
            "Last_ACC": last_acc,
            "Last_Kappa": last_kappa,
            "noise_type": "white_noise",
            "noise_params": noise_c,
            "dataset": data_set_name,
        }
        with open(f'{destination_path}ann_white_noise_{white_noise_strength}_test_description.json', 'w') as exp_file:
            json.dump(ann_test_results, exp_file)

        best_acc, best_kappa, last_acc, last_kappa = eval_snn_model(
            **snn_eval_config
        )

        snn_test_results = {
            "ModelType": "SNN",
            "Best_ACC": best_acc,
            "Best_Kappa": best_kappa,
            "Last_ACC": last_acc,
            "Last_Kappa": last_kappa,
            "noise_type": "white_noise",
            "noise_params": noise_c,
            "dataset": data_set_name,
        }
        with open(f'{destination_path}snn_white_noise_{white_noise_strength}_test_description.json', 'w') as exp_file:
            json.dump(snn_test_results, exp_file)

        # Advanced save path is Experiments Dataset_test_whitenoise/noise_c_models.json
        # Save each run single and also collect to save in one file at the end per dataset
    end = time.time()
    print(f"Last run took {end-start}")

    # Save all stats
    # Generate Graph for ANN and noise to eval
    # Generate Graph for SNN and noise to eval


# when done with all models, create one file with all values.json from all experiments gdf