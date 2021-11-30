from run_basic_training_asci import main as run_training
import math

learning_rates = [0.01,0.005,0.001,0.0005,0.0001]
weight_decays = [0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
cut_offs = [[0,200],[25,225],[50,250],[75,275],[100,300]]
dropouts = [0.1,0.3,0.5,0.7]

best_params = []
if __name__ == '__main__':
    subjects = [['k3b','1'],['k6b','2'],['l1b','3']]

    for s in subjects:
        overall_best_accuracy = -math.inf
        params = None
        for learning_rate in learning_rates:
            for weight_decay in weight_decays:
                for dropout in dropouts:
                    for cut_off in cut_offs:
                        cut_off_front = cut_off[0]
                        cut_off_back = cut_off[1]
                        experiment_name = f'Binary_ANN_{s[0]}_learnrate{learning_rate}_weightdec{weight_decay}_cutoff{cut_off_front}_{cut_off_back}_drop{dropout}'
                        experiment_description = 'One vs Rest classification of ASCI based motor imagery classifcation on ANN Architecture. ' \
                                                 'Applies Error averaging, CSP, Normalization and then transforms into CWT'
                        best_acc, last_acc = run_training(experiment_name, experiment_description,
                                                          learning_rate, weight_decay, cut_off_front, cut_off_back,
                                                          dropout, subject_name=s[0], subject_nr=s[1])
                        if best_acc > overall_best_accuracy or last_acc > overall_best_accuracy:
                            overall_best_accuracy = max(best_acc, last_acc)
                            params = {'learning_rate': learning_rate
                                , 'weight_decay': weight_decay
                                , 'cut_off_front': cut_off_front
                                , 'cut_off_back': cut_off_back
                                , 'dropout': dropout
                                , 'best_accuracy_multi': max(best_acc, last_acc)}
                            print('New best Config Found')
                            print(params)
        best_params.append([f'ANN_{s[0]}', params])

    with open('ANN_ASCI_HyperResults.txt', 'w+') as final_file:
        for r in best_params:
            params = r[1]
            final_file.write(f'Subject: {r[0]}, learning_rate: {params["learning_rate"]}, '
                             f'weight_decay:{params["weight_decay"]}, cut_off_front: {params["cut_off_front"]}, '
                             f'cut_off_back: {params["cut_off_back"]}, dropout: {params["dropout"]}, '
                             f'ACC: {params["best_accuracy_multi"]}/n/n')