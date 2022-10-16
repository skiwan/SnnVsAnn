import os, shutil
from Utils import create_temp_folder
from load_eeg_from_GDF import load_eeg_from_gdf
from apply_CSP import apply_csp
from normalize_feature_extraction import apply_normlized_feature_extraction
from apply_CWT import apply_cwt
from Models.data_splitter import DataSplitter
from noise_configs import burst_noise_values
from apply_burst_noise import apply_burst_noise
from Utils import create_folder_if_not_exists
import json
import time




from training_helper_gdf import train_model_ann, train_model_snn, eval_ann_model, eval_snn_model

from model_configs import base_model_config, gdf_models
# Prep

file_directory = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(file_directory)
base_save_path = os.path.join(file_directory, 'temp/')

# Create temp folder if not exists
create_temp_folder(file_directory)

for data_set_name, model_c in gdf_models.items():
    start = time.time()

    train_file_name = f'{data_set_name}T.gdf'
    eval_file_name = f'{data_set_name}E.gdf'
    eval_label_file_name = f'{data_set_name}E_labels.npy'


    base_file_path = os.path.join(parent_folder, 'original_data/Datasets/BCICompetitionIV/Data/BCICIV_2a_gdf/')
    base_label_path = os.path.join(parent_folder, 'savecopywithlabels/')

    raw_train_file_name = os.path.join(base_file_path, train_file_name)
    raw_eval_file_name = os.path.join(base_file_path, eval_file_name)

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
                                     f'{base_save_path}normalized_train_class{i}_labels.npy', f'{base_save_path}snn_class{i}',
                                     split_ratio)
        data_splitter.split(splitting_strategy)

        # Prepare Train and Val sets ANN
        data_splitter = DataSplitter(f'{base_save_path}cwt_train_class{i}.npy',
                                     f'{base_save_path}normalized_train_class{i}_labels.npy', f'{base_save_path}ann_class{i}',
                                     split_ratio)
        data_splitter.split(splitting_strategy)

    experiment_description = f"Burst Noise Test Data Set: {data_set_name} base Truth"

    experiment_config_base = {
        "base_save_path": base_save_path,
        "file_directory": file_directory,
        "train_file_name": train_file_name,
        "eval_file_name": eval_file_name,
        "experiment_file_name": f"{data_set_name}_basetrain_burstnoise",
        "device": "cpu",
    }

    # Prepare SNN config
    ann_config = {
        "experiment_description": f"{experiment_description} ANN",
        "experiment_name": f"{data_set_name}_basetrain_burstnoise_ann",
    }
    ann_config.update(experiment_config_base)
    ann_config.update(base_model_config)
    ann_config.update(model_c["ANN"])
    # Train ANN on normal data
    ann_pre = train_model_ann(ann_config)



    snn_config = {
        "experiment_description": f"{experiment_description} SNN",
        "experiment_name": f"{data_set_name}_basetrain_burstnoise_snn",
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
        destination_path, f"{data_set_name}_burstnoise_test"
    )
    create_folder_if_not_exists(destination_path)
    destination_path = f'{destination_path}/'

    ann_eval_config = {'base_save_path': ann_config['base_save_path'], 'experiment_name': ann_config['experiment_name'],
                       'model_channels': ann_config['model_channels'], 'model_classes': ann_config['model_classes'],
                       'model_dropout': ann_config['dropout'], "data_cut_front": model_c["ANN"]["cut_off"][0],
                       "data_cut_back": model_c["ANN"]["cut_off"][1]}

    snn_eval_config = {'base_save_path': snn_config['base_save_path'], 'experiment_name': snn_config['experiment_name'],
                       'model_channels': snn_config['model_channels'], 'model_classes': snn_config['model_classes']}

    for noise_c in burst_noise_values:
        # Generate noisey Data
        load_eeg_from_gdf(low_pass, high_pass, raw_train_file_name, f'{base_save_path}raw_train', frequency=250,
                          trial_duration=6, noise_fn=apply_burst_noise, noise_fn_params=noise_c)
        load_eeg_from_gdf(low_pass, high_pass, raw_eval_file_name, f'{base_save_path}raw_eval', frequency=250,
                          trial_duration=6, noise_fn=apply_burst_noise, noise_fn_params=noise_c)

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

        burst_noise_strength = noise_c["noise_strength_percent"]
        burst_burst_amounts = noise_c["burst_amounts"]

        ann_test_results = {
            "ModelType": "ANN",
            "Best_ACC": best_acc,
            "Best_Kappa": best_kappa,
            "Last_ACC": last_acc,
            "Last_Kappa": last_kappa,
            "noise_type": "burst_noise",
            "noise_params": noise_c,
            "dataset": data_set_name,
        }
        with open(f'{destination_path}ann_burst_noise_{burst_noise_strength}_{burst_burst_amounts}_test_description.json', 'w') as exp_file:
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
            "noise_type": "burst_noise",
            "noise_params": noise_c,
            "dataset": data_set_name,
        }
        with open(f'{destination_path}snn_burst_noise_{burst_noise_strength}_{burst_burst_amounts}_test_description.json', 'w') as exp_file:
            json.dump(snn_test_results, exp_file)

        # Advanced save path is Experiments Dataset_test_burstnoise/noise_c_models.json
        # Save each run single and also collect to save in one file at the end per dataset
    end = time.time()
    print(f"Last run took {end-start}")


    # Save all stats
    # Generate Graph for ANN and noise to eval
    # Generate Graph for SNN and noise to eval


# when done with all models, create one file with all values.json from all experiments gdf