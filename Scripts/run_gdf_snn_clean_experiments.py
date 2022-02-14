from run_basic_training_snn_gdf import main as run_training
import sys

learning_rates = [0.01,0.005,0.001,0.0005,0.0001]
weight_decays = [0,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
cut_offs = [[0,200],[25,225],[50,250],[75,275],[100,300]]
#nr = ['01', '02', '03', '05', '06', '07', '08', '09']
nr = ['08']

def main(start_expriment_nr=0,max_gpus=1, process_per_gpu=1):
    start_expriment_nr = int(start_expriment_nr)
    max_gpus = int(max_gpus)
    process_per_gpu=int(process_per_gpu)
    best_params = [None for x in range(len(nr))]


    for i, n in enumerate(nr):
        result_collector = {}
        experiment_name = f'Binary_SNN_A{n}'
        experiment_description = 'One vs Rest classification of GDF based motor imagery classifcation on SNN Architecture. ' \
                                 'Applies Error averaging, CSP and Normalization'
        train_file_name = f'A{n}T.gdf'
        eval_file_name = f'A{n}E.gdf'
        eval_label_file_name = f'A{n}E_labels.npy'
        device = f'cuda'
        best_params_s = run_training(experiment_name, experiment_description,
                                     train_file_name, eval_file_name, eval_label_file_name,
                                     learning_rates, weight_decays, cut_offs,
                                     start_expriment_nr, device, max_gpus, process_per_gpu)

        best_params[i] = best_params_s

    with open('SNN_GDF_HyperResults_8.txt', 'w+') as final_file:
        for i, r in enumerate(best_params):
            final_file.write(f'Subject: {nr[i]}, best_conf: {r}/n/n)')

    print(best_params)


if __name__ == '__main__':
    main(*sys.argv[1:])
