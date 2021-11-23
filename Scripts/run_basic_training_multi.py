from Models.data_combiner import DataCombiner
from Models.data_splitter import DataSplitter
from Utils import create_temp_folder, delete_temp_folder, create_folder_if_not_exists
from load_eeg_from_GDF import load_eeg_from_gdf
from apply_CSP import apply_csp
from normalize_feature_extraction import apply_normlized_feature_extraction
from apply_CWT import apply_cwt
from Scripts.multi_class_ann_run import run_multiclass_classification, load_and_run_eval
import shutil
import os, sys
import json

def main(experiment_name, experiment_description, train_file_name, eval_file_name, eval_label_file_name):
    file_directory = os.path.dirname(os.path.abspath(__file__))
    base_save_path = os.path.join(file_directory, 'temp/')

    # Create temp folder if not exists
    create_temp_folder(file_directory)

    experiment_name = experiment_name #'Binary_ANN_A01'
    experiment_description = experiment_description# 'Something Something'
    train_file_name = train_file_name# 'A01T.gdf'
    eval_file_name = eval_file_name# 'A01E.gdf'
    eval_label_file_name = eval_label_file_name#'A01E_labels.npy'

    base_file_path = '/home/merlinsewina/MaWork/SnnVsAnn/original_data/Datasets/BCICompetitionIV/Data/BCICIV_2a_gdf/'

    base_label_path = '/home/merlinsewina/MaWork/SnnVsAnn/savecopywithlabels/'
    low_pass = 7
    high_pass = 30
    raw_train_file_name = os.path.join(base_file_path, train_file_name)
    raw_eval_file_name = os.path.join(base_file_path, eval_file_name)
    extract_features = False
    scale_1 = [7,15,0.5]
    scale_2 = [16,30,0.5]
    split_ratio = 0.7
    splitting_strategy = 'balanced'
    batch_size = 8
    shuffle = True
    workers = 3
    max_epochs = 100
    model_channels = 32
    model_classes = 4
    model_dropout = 0.3
    model_learning_rate = 0.001
    model_weight_decay = 0.0001
    data_cut_front = 50
    data_cut_back = 250
    save_model = True


    destination_path = os.path.join(file_directory, 'Experiments')
    destination_path = os.path.join(destination_path, f'{experiment_name}')
    create_folder_if_not_exists(destination_path)
    destination_path = f'{destination_path}/'

    ## Save parameters, input settings and experiment description in script string
    experiment_setup_info = {'Experiment_Name': experiment_name}
    experiment_setup_info['Description'] = experiment_description
    experiment_setup_info['Experiment Type'] = 'Multiclass'
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
    train_dataset_paths = []
    eval_datset_paths = []
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
        train_dataset_paths.append(f'{base_save_path}cwt_train_class{i}.npy')

        apply_cwt(f'{base_save_path}normalized_eval_class{i}.npy', f'{base_save_path}cwt_eval_class{i}.npy',
                  scale_1[0], scale_1[1], scale_1[2], scale_2[0], scale_2[1], scale_2[2])
        eval_datset_paths.append(f'{base_save_path}cwt_eval_class{i}.npy')


    # Combine all 4 subclasses into 1 Dataset
    data_combiner = DataCombiner(train_dataset_paths, f'{base_save_path}raw_train_labels.npy', f'{base_save_path}_train')
    data_combiner.combine()

    data_combiner = DataCombiner(eval_datset_paths, f'{base_save_path}raw_eval_labels.npy', f'{base_save_path}_eval')
    data_combiner.combine()

    # Prepare Train and Val sets
    data_splitter = DataSplitter(f'{base_save_path}_train_whole_set.npy', f'{base_save_path}_train_whole_labels.npy', f'{base_save_path}_whole', split_ratio)
    data_splitter.split(splitting_strategy)

    # Load binary model script with parameters
    # Run Training and additionally save best val model as well as its epoch of origin
    statistics, e_loss, eval_acc, eval_kappa, best_val_epoch = run_multiclass_classification(
        batch_size, shuffle, workers, max_epochs,
        f'{base_save_path}_whole_train_data.npy', f'{base_save_path}_whole_train_labels.npy',
        f'{base_save_path}_whole_validate_data.npy', f'{base_save_path}_whole_validate_labels.npy',
        f'{base_save_path}_eval_whole_set.npy', f'{base_save_path}_eval_whole_labels.npy',
        model_channels, model_classes, model_dropout,
        model_learning_rate, model_weight_decay, data_cut_front, data_cut_back, save_model, f'{base_save_path}{experiment_name}_whole_model'
    )
    experiment_setup_info[f'Whole_Evaluation_Loss'] = e_loss.item()
    experiment_setup_info[f'Whole_Evaluation_Accuracy'] = eval_acc
    experiment_setup_info[f'Whole_Evaluation_Kappa'] = eval_kappa
    experiment_setup_info[f'Whole_Best_Validation_Epoch'] = best_val_epoch

    # Save Training Run Statistics
    train_statistics = {
        'epoch' : statistics[:][0]
        ,'train_loss': statistics[:][1]
        ,'train_acc': statistics[:][2]
        ,'val_loss': statistics[:][3]
        ,'val_acc': statistics[:][4]
    }
    ## save statistic of training
    with open(f'{destination_path}train_statistics_whole.json', 'w') as stats_file:
        json.dump(train_statistics, stats_file)

    # If feature extraction is True move filters file
    #TODO change save name to include xperiment name and class for binary (noth files)
    if extract_features:
        shutil.copyfile(f'{base_save_path}normalized_train_filters.npy', f'{destination_path}normalized_train_filters.npy')

    # load saved best val model and apply eval set
    if save_model:
        e_loss, eval_acc, eval_kappa = load_and_run_eval(
            f'{base_save_path}{experiment_name}_whole_model.pth'
            , f'{base_save_path}_eval_whole_set.npy', f'{base_save_path}_eval_whole_labels.npy'
            ,data_cut_front, data_cut_back, model_channels, model_classes, model_dropout)
        ## save statistic of eval on best val model
        experiment_setup_info[f'Best_Model_Whole_Evaluation_Loss'] = e_loss.item()
        experiment_setup_info[f'Best_Model_Whole_Evaluation_Accuracy'] = eval_acc
        experiment_setup_info[f'Best_Model_Whole_Evaluation_Kappa'] = eval_kappa


    # Save The experiment setup String in Experiments folder
    with open(f'{destination_path}experiment_description.json', 'w') as exp_file:
        json.dump(experiment_setup_info, exp_file)
    # Delete the temp Folder
    delete_temp_folder(file_directory)

if __name__ == "__main__":
	main(*sys.argv[1:])
