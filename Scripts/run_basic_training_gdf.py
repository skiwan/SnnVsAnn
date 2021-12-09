from Models.data_splitter import DataSplitter
from Utils import create_temp_folder, delete_temp_folder, create_folder_if_not_exists, delete_folder
from load_eeg_from_GDF import load_eeg_from_gdf
from apply_CSP import apply_csp
from normalize_feature_extraction import apply_normlized_feature_extraction
from apply_CWT import apply_cwt
from Scripts.binary_class_ann_run import run_binary_classification, load_and_run_eval
from Scripts.multi_class_ann_run import main as multiclass_run
import shutil
import os, sys
import json
import math
import threading
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch

import logging
logging.basicConfig(level=logging.DEBUG)


def run_threaded_model(config):
    try:
        base_save_path = config['base_save_path']
        batch_size = config['batch_size']
        cut_off = config['cut_off']
        device = config['device']
        dropout = config['dropout']
        eval_file_name = config['eval_file_name']
        experiment_description = config['experiment_description']
        experiment_name = config['experiment_name']
        extract_features = config['extract_features']
        file_directory = config['file_directory']
        high_pass = config['high_pass']
        learning_rate = config['learning_rate']
        low_pass = config['low_pass']
        max_epochs = config['max_epochs']
        model_channels = config['model_channels']
        model_classes = config['model_classes']
        scale_1 = config['scale_1']
        scale_2 = config['scale_2']
        shuffle = config['shuffle']
        split_ratio = config['split_ratio']
        splitting_strategy = config['splitting_strategy']
        train_file_name = config['train_file_name']
        weight_decay = config['weight_decay']
        workers = config['workers']

        model_dropout = dropout
        model_learning_rate = learning_rate
        model_weight_decay = weight_decay
        data_cut_front = cut_off[0]
        data_cut_back = cut_off[1]
        experiment_name = f'{experiment_name}_learnrate{learning_rate}_weightdec{weight_decay}_cutoff{data_cut_front}_{data_cut_back}_drop{dropout}'
        save_model = True
        destination_path = os.path.join(file_directory, 'Experiments')
        destination_path = os.path.join(destination_path, f'{experiment_name}')
        create_folder_if_not_exists(destination_path)
        destination_path = f'{destination_path}/'
        ## Save parameters, input settings and experiment description in script string
        experiment_setup_info = {'Experiment_Name': experiment_name}
        experiment_setup_info['Description'] = experiment_description
        experiment_setup_info['Experiment Type'] = 'One-vs-Rest'
        experiment_setup_info['Base_Train_file'] = train_file_name
        experiment_setup_info['Base_Eval_file'] = eval_file_name
        experiment_setup_info['Base_Format'] = 'GDF'
        experiment_setup_info['Frequency_Bands'] = [[low_pass, high_pass]]
        experiment_setup_info['Feature Extraction'] = extract_features
        experiment_setup_info['CWT_Scales'] = [scale_1, scale_2]
        experiment_setup_info['Train_Val_Ratio'] = split_ratio
        experiment_setup_info['Splitting_Strategy'] = splitting_strategy
        experiment_setup_info['Batch_Size'] = batch_size
        experiment_setup_info['Shuffle_Data'] = shuffle
        experiment_setup_info['Load_Workers'] = workers
        experiment_setup_info['Train_Epoch_Amount'] = max_epochs
        experiment_setup_info['Model_Channels'] = model_channels
        experiment_setup_info['Model_Classes'] = model_classes
        experiment_setup_info['Dropout_Value'] = model_dropout
        experiment_setup_info['Learning_Rate'] = model_learning_rate
        experiment_setup_info['Weight_Decay'] = model_weight_decay
        experiment_setup_info['Sample_Data_Start'] = data_cut_front
        experiment_setup_info['Sample_Data_End'] = data_cut_back
        experiment_setup_info['Save_Best_Models'] = save_model
        for i in range(1, 5):
            # Load binary model script with parameters
            # Run Training and additionally save best val model as well as its epoch of origin
            statistics, e_loss, eval_acc, eval_kappa, best_val_epoch = run_binary_classification(
                batch_size, shuffle, workers, max_epochs,
                f'{base_save_path}_class{i}_train_data.npy', f'{base_save_path}_class{i}_train_labels.npy',
                f'{base_save_path}_class{i}_validate_data.npy', f'{base_save_path}_class{i}_validate_labels.npy',
                f'{base_save_path}cwt_eval_class{i}.npy', f'{base_save_path}normalized_eval_class{i}_labels.npy',
                model_channels, model_classes, model_dropout,
                model_learning_rate, model_weight_decay, data_cut_front, data_cut_back, save_model,
                f'{base_save_path}{experiment_name}_class{i}_model', device
            )
            experiment_setup_info[f'Class_{i}_Evaluation_Loss'] = e_loss.item()
            experiment_setup_info[f'Class_{i}_Evaluation_Accuracy'] = eval_acc
            experiment_setup_info[f'Class_{i}_Evaluation_Kappa'] = eval_kappa.item()
            experiment_setup_info[f'Class_{i}_Best_Validation_Epoch'] = best_val_epoch

            # Save Training Run Statistics
            train_statistics = {
                'epoch': statistics[0]
                , 'train_loss': statistics[1]
                , 'train_acc': statistics[2]
                , 'val_loss': statistics[3]
                , 'val_acc': statistics[4]
            }
            ## save statistic of training
            with open(f'{destination_path}train_statistics_class{i}.json', 'w') as stats_file:
                json.dump(train_statistics, stats_file)

            # If feature extraction is True move filters file
            if extract_features:
                shutil.copyfile(f'{base_save_path}normalized_train_filters.npy',
                                f'{destination_path}normalized_train_filters.npy')

            # load saved best val model and apply eval set
            if save_model:
                e_loss, eval_acc, eval_kappa = load_and_run_eval(
                    f'{base_save_path}{experiment_name}_class{i}_model.pth'
                    , f'{base_save_path}cwt_eval_class{i}.npy', f'{base_save_path}normalized_eval_class{i}_labels.npy'
                    , data_cut_front, data_cut_back, model_channels, model_classes, model_dropout, device)
                ## save statistic of eval on best val model
                experiment_setup_info[f'Best_Model_Class_{i}_Evaluation_Loss'] = e_loss.item()
                experiment_setup_info[f'Best_Model_Class_{i}_Evaluation_Accuracy'] = eval_acc
                experiment_setup_info[f'Best_Model_Class_{i}_Evaluation_Kappa'] = eval_kappa.item()
        # call the multiclass function on all 4 models
        best_acc, best_kappa, last_acc, last_kappa = multiclass_run(base_save_path, experiment_name, 4
                                                                    , data_cut_front, data_cut_back, model_channels,
                                                                    model_classes, model_dropout, 'cpu')
        experiment_setup_info[f'MultiClass_Best_Evaluation_Accuracy'] = best_acc
        experiment_setup_info[f'MultiClass_Best_Evaluation_Kappa'] = best_kappa
        experiment_setup_info[f'MultiClass_Last_Evaluation_Accuracy'] = last_acc
        experiment_setup_info[f'MultiClass_Last_Evaluation_Kappa'] = last_kappa
        print(f'Multiclass Accuarcy best: {best_acc}, last: {last_acc}')
        # Save The experiment setup String in Experiments folder
        with open(f'{destination_path}experiment_description.json', 'w') as exp_file:
            json.dump(experiment_setup_info, exp_file)
        overall_best_acc = max(best_acc, last_acc)
        tune.report(mean_accuracy=overall_best_acc)
        #return best_acc, experiment_setup_info, last_acc
        return overall_best_acc
    except Exception as e:
        logging.exception('HELP> RUN INTO AN ERROR') # log exception info at CRITICAL log level



def main(experiment_name, experiment_description, train_file_name, eval_file_name, eval_label_file_name,
         learning_rates, weight_decays, cut_offs, dropouts,
         experiment_number=0, device='cuda', max_gpus=1, process_per_gpu=1):
    file_directory = os.path.dirname(os.path.abspath(__file__))
    parent_folder = os.path.dirname(file_directory)
    base_save_path = os.path.join(file_directory, 'temp/')

    # Create temp folder if not exists
    create_temp_folder(file_directory)

    experiment_description = experiment_description
    train_file_name = train_file_name
    eval_file_name = eval_file_name
    eval_label_file_name = eval_label_file_name

    base_file_path = os.path.join(parent_folder, 'original_data/Datasets/BCICompetitionIV/Data/BCICIV_2a_gdf/')
    base_label_path = os.path.join(parent_folder, 'savecopywithlabels/')

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
    max_epochs = 100
    model_channels = 8
    model_classes = 2


    ## DATA PREP START
    # load raw file into temp folder
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
    for i in range(1,5):
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

        # Prepare Train and Val sets
        data_splitter = DataSplitter(f'{base_save_path}cwt_train_class{i}.npy', f'{base_save_path}normalized_train_class{i}_labels.npy', f'{base_save_path}_class{i}', split_ratio)
        data_splitter.split(splitting_strategy)
    ## DATA PREP END

    ## Threaded Model Start

    max_gpus = int(max_gpus)
    process_per_gpu=int(process_per_gpu)

    device = 'cuda'
    ray_config = {
        'base_save_path': base_save_path,
        'batch_size': batch_size,
        'cut_off': tune.grid_search(cut_offs),
        'device': device,
        'dropout': tune.grid_search(dropouts),
        'eval_file_name': eval_file_name,
        'experiment_description': experiment_description,
        'experiment_name': experiment_name,
        'extract_features': extract_features,
        'file_directory': file_directory,
        'high_pass': high_pass,
        'learning_rate': tune.grid_search(learning_rates),
        'low_pass': low_pass,
        'max_epochs': max_epochs,
        'model_channels': model_channels,
        'model_classes': model_classes,
        'scale_1': scale_1,
        'scale_2': scale_2,
        'shuffle': shuffle,
        'split_ratio': split_ratio,
        'splitting_strategy': splitting_strategy,
        'train_file_name': train_file_name,
        'weight_decay': tune.grid_search(weight_decays),
        'workers': workers
    }


    analysis = tune.run(
        run_threaded_model,
        config=ray_config, resources_per_trial={"gpu": 1/process_per_gpu}, num_samples=1, mode='max', max_concurrent_trials=max_gpus * process_per_gpu)

    print("Best config: ", analysis.get_best_config(
        metric="mean_accuracy", mode="max"))
    # Delete the temp Folder
    delete_temp_folder(file_directory)
    return ''


if __name__ == "__main__":
	main(*sys.argv[1:])
