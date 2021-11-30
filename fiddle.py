from Scripts.multi_class_ann_run import main as multiclass_run
import os



file_directory = os.path.dirname(os.path.abspath(__file__))
base_save_path = os.path.join(os.path.join(file_directory, 'Scripts'), 'temp/')
experiment_name = 'Binary_ANN_A01'
data_cut_front = 50
data_cut_back = 250
model_channels = 8
model_classes = 2
model_dropout = 0.3
device = 'cpu'



multiclass_run(base_save_path, experiment_name, 4
         ,data_cut_front, data_cut_back, model_channels, model_classes, model_dropout, device)



