base_model_config = {
    "low_pass" : 7,
    "high_pass" : 30,
    "extract_features" : False,
    "scale_1" : [7, 15, 0.5],
    "scale_2" : [16, 30, 0.5],
    "split_ratio" : 0.7,
    "splitting_strategy" : 'balanced-copy',
    "batch_size" : 32,
    "shuffle" : True,
    "workers" : 1,
    "max_epochs" : 300,
    "model_channels" : 8,
    "model_classes" : 2,
}

gdf_models = {}


asci_models = {}