from run_basic_training_asci import main as run_training
import math
import sys
import threading

learning_rates = [0.01,0.005,0.001,0.0005,0.0001]
weight_decays = [0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
cut_offs = [[0,200],[25,225],[50,250],[75,275],[100,300]]
dropouts = [0.1,0.3,0.5,0.7]
subjects = [['k3b','1'],['k6b','2'],['l1b','3']]
subject_experiments_amount = len(learning_rates) * len(weight_decays) * len(cut_offs) * len(dropouts)
total_experiments_amount = subject_experiments_amount * len(subjects)

def main(start_expriment_nr=0,max_gpus=1, process_per_gpu=1, subject_experiments_amount=subject_experiments_amount, total_experiments_amount=total_experiments_amount):
    current_experiment_nr = start_expriment_nr
    max_threads = max_gpus * process_per_gpu
    current_threads = []
    best_params = [None for x in range(len(subjects))]
    overall_best_accuracy = [-math.inf for i in range(len(subjects))]

    result_collector = {}

    while current_experiment_nr < total_experiments_amount:
        if len(current_threads) < max_threads:
            s = subjects[current_experiment_nr//subject_experiments_amount]
            learning_rate = learning_rates[current_experiment_nr //(len(cut_offs)*len(dropouts)*len(weight_decays)) % len(learning_rates)] # fourth
            weight_decay = weight_decays[current_experiment_nr//( len(cut_offs)*len(dropouts)) % len(weight_decays)] # third
            dropout = dropouts[current_experiment_nr // (len(cut_offs)) % len(dropouts)] # second
            cut_off = cut_offs[current_experiment_nr % len(cut_offs)] # first
            cut_off_front = cut_off[0]
            cut_off_back = cut_off[1]
            experiment_name = f'Binary_ANN_{s[0]}_learnrate{learning_rate}_weightdec{weight_decay}_cutoff{cut_off_front}_{cut_off_back}_drop{dropout}'
            experiment_description = 'One vs Rest classification of ASCI based motor imagery classifcation on ANN Architecture. ' \
                                     'Applies Error averaging, CSP, Normalization and then transforms into CWT'
            c_params = {'learning_rate': learning_rate
                                , 'weight_decay': weight_decay
                                , 'cut_off_front': cut_off_front
                                , 'cut_off_back': cut_off_back
                                , 'dropout': dropout
                        #        , 'best_accuracy_multi': max(best_acc, last_acc)
                        }
            # result_collector key is experiment number, added is a dic with best_acc, last_acc, c_params
            device = f'cuda:{len(current_threads)//process_per_gpu}'

            result_collector[current_experiment_nr] = {'c_params': c_params}
            current_threads.append(threading.Thread(target=run_training,
                                            args=(experiment_name, experiment_description,
                                            learning_rate, weight_decay, cut_off_front, cut_off_back,
                                            dropout,s[0], s[1],
                                            result_collector, current_experiment_nr, device)))
            current_experiment_nr += 1
        else:
            for prepared_thread in current_threads:
                prepared_thread.start()
            for prepared_thread in current_threads:
                prepared_thread.join()
            current_threads = []


    for i, exp_key in enumerate(list(result_collector.keys())):
        exp_dic = result_collector[exp_key]
        if exp_dic['best_acc'] > overall_best_accuracy[i//subject_experiments_amount] or exp_dic['last_acc'] > overall_best_accuracy[i//subject_experiments_amount]:
            overall_best_accuracy[i // subject_experiments_amount] = max(exp_dic['best_acc'], exp_dic['last_acc'])
            params = exp_dic['c_params']
            params['best_accuracy_multi'] = overall_best_accuracy[i // subject_experiments_amount]
            print(f'New best Config Found for subject {i // subject_experiments_amount}')
            print(params)
            best_params[i // subject_experiments_amount] = [f'ANN_{s[0]}', params]


    with open('ANN_ASCI_HyperResults.txt', 'w+') as final_file:
        for r in best_params:
            params = r[1]
            final_file.write(f'Subject: {r[0]}, learning_rate: {params["learning_rate"]}, '
                             f'weight_decay:{params["weight_decay"]}, cut_off_front: {params["cut_off_front"]}, '
                             f'cut_off_back: {params["cut_off_back"]}, dropout: {params["dropout"]}, '
                             f'ACC: {params["best_accuracy_multi"]}/n/n')

if __name__ == '__main__':
    main(*sys.argv[1:])


