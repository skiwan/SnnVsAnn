from run_basic_training_multi_asci import main as run_training


if __name__ == '__main__':
    subjects = [['k3b','1'],['k6b','2'],['l1b','3']]

    for s in subjects:
        experiment_name = f'Multi_ANN_{s[0]}'
        experiment_description = 'Multiclass classification of ASCI based motor imagery classifcation on ANN Architecture. ' \
                                 'Applies Error averaging, CSP, Normalization and then transforms into CWT'
        run_training(experiment_name, experiment_description, subject_name=s[0], subject_nr=s[1])