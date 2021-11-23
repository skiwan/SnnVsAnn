from run_basic_training_binary import main as run_training


if __name__ == '__main__':
    nr = ['01', '02', '03', '05', '06', '07', '08', '09']

    for n in nr:
        experiment_name = f'Binary_ANN_A{n}'
        experiment_description = 'One vs Rest classification of GDF based motor imagery classifcation on ANN Architecture. ' \
                                 'Applies Error averaging, CSP, Normalization and then transforms into CWT'
        train_file_name = f'A{n}T.gdf'
        eval_file_name = f'A{n}E.gdf'
        eval_label_file_name = f'A{n}E_labels.npy'
        run_training(experiment_name, experiment_description, train_file_name, eval_file_name, eval_label_file_name)