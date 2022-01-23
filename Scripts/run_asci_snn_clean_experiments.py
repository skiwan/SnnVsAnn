from run_basic_training_asci import main as run_training
import sys

learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]
weight_decays = [0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
cut_offs = [[0, 200], [25, 225], [50, 250], [75, 275], [100, 300]]
subjects = [['k3b', '1'], ['k6b', '2'], ['l1b', '3']]


def main(start_expriment_nr=0, max_gpus=1, process_per_gpu=1):
    start_expriment_nr = int(start_expriment_nr)
    max_gpus = int(max_gpus)
    process_per_gpu = int(process_per_gpu)
    best_params = [None for x in range(len(subjects))]

    for i, s in enumerate(subjects):
        experiment_name = f'Binary_SNN_{s[0]}'
        experiment_description = 'One vs Rest classification of ASCI based motor imagery classifcation on SNN Architecture. ' \
                                 'Applies Error averaging, CSP and Normalization'
        device = f'cuda'
        best_params_s = run_training(experiment_name, experiment_description,
                                     learning_rates, weight_decays, cut_offs,
                                     s[0], s[1],
                                     start_expriment_nr, device, max_gpus, process_per_gpu)
        best_params[i] = best_params_s

    with open('SNN_ASCI_HyperResults.txt', 'w+') as final_file:
        for i, r in enumerate(best_params):
            final_file.write(f'Subject: {subjects[i][0]}, best_conf: {r}/n/n)')


if __name__ == '__main__':
    main(*sys.argv[1:])
