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
    "max_epochs" : 1,
    "model_channels" : 8,
    "model_classes" : 2,
}

gdf_models = {
    "A01" : {
        "SNN" : {"learning_rate": 0.01, "weight_decay": 0.0},
        "ANN" : {"learning_rate": 0.001, "weight_decay": 0.1, "cut_off": [50,250], "dropout": 0.3},
    },
    "A02" : {},
    "A03" : {},
    "A05" : {},
    "A06" : {},
    "A07" : {},
    "A08" : {},
    "A09" : {},
}


asci_models = {}