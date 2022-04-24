import os
import numpy as np
import matplotlib.pyplot as plt
from norse.torch.utils.plot.plot import plot_spikes_2d

file_directory = os.path.dirname(os.path.abspath(__file__))
file_directory = os.path.join(file_directory, 'Scripts/')
base_save_path = os.path.join(file_directory, 'temp/')
# label, class samples

model_types = ["best", "last"]


for label in range(4):
    for model_type in model_types:
        interim_data = np.load(f'_{model_type}_models_label_{label}_inter_activity.npy', allow_pickle=True)
        output_data = np.load(f'_{model_type}_models_label_{label}_activity.npy', allow_pickle=True)
        # PLot position -> Data
        datas = {"interim": [0, interim_data], "output": [1, output_data]}

        samples = len(interim_data[0])

        for sample in range(samples):
            main_fig, main_plots = plt.subplots(4,2)
            main_fig.set_size_inches(32, 12)

            for model in range(4):
                sub_fig, sub_plots = plt.subplots(2,1)
                s = 0
                for data_type in datas.keys():
                    current_data = datas[data_type][1]
                    current_col = datas[data_type][0]
                    sample_data = current_data[model][sample]

                    plot_col = model % 2
                    plot_row = (model//2)*2 + current_col


                    current_plot = main_plots[plot_row,plot_col]

                    for t_plot in [current_plot, sub_plots[s]]:
                        t_plot.set_title(
                            f'Label {label + 1} - Model {model} {model_type} - {data_type} Sample {sample}')
                        t_plot.set_xlabel("Timestep")
                        t_plot.set_ylabel("Neuron")
                        t_plot.set_xlim([0, 1500])

                    full_y = sample_data.squeeze().tolist()
                    neurons = len(full_y[0])
                    legend = []
                    for neuron in range(neurons):
                        x = [i for i in range(1500) if full_y[i][neuron] > 0]
                        y = [full_y[i][neuron] for i in x]
                        y = [i*(neuron+1)-1 for i in y]
                        sub_plots[s].scatter(x, y)
                        current_plot.scatter(x, y)
                        if data_type == "output":
                            legend.append(len(y))

                    if data_type == "output":
                        sub_plots[s].legend([f"Yes Neuron {legend[0]}", f"No Neuron {legend[1]}"])
                        current_plot.legend([f"Yes Neuron {legend[0]}", f"No Neuron {legend[1]}"])
                    s += 1
                sub_fig.savefig(f'{model_type}_label{label + 1}_model{model + 1}_sample_{sample}_activity.png', dpi=200)
                plt.clf()
            main_fig.savefig(f'{model_type}_label{label + 1}_all_models_sample_{sample}_activity.png', dpi=200)
            main_fig.clf()


# Load either last or best
    # For all samples
        # Prepare subplot of size 2, models
        # FOr all models
            # Load either inter our out
                # Get SUbplot for this part
                # Load sample for that data
                # for each neuron in that sample
                # Create a subfig on that plot with the output, a color randomly
                # Set title axis legends etc on SUbplot
        # Save


"""
    interim , interim
    output, output
    interim, interim
    output, output
    column rows figurenr
    #plt.subplot(4,2,)
    
    +
    
    interim
    output
    sublot (1,2)
"""

