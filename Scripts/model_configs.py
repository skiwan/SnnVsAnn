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

gdf_models = {
    "A01" : {
        "SNN" : {"learning_rate": 0.01, "weight_decay": 0.0},
        "ANN" : {"learning_rate": 0.001, "weight_decay": 0.1, "cut_off": [50,250], "dropout": 0.3},
    },
    "A02" : {
        "SNN" : {"learning_rate": 0.005, "weight_decay": 0.0},
        "ANN" : {"learning_rate": 0.001, "weight_decay": 0.05, "cut_off": [100,300], "dropout": 0.1},
    },
    "A03" : {
        "SNN" : {"learning_rate": 0.01, "weight_decay": 0.0},
        "ANN" : {"learning_rate": 0.0001, "weight_decay": 0.5, "cut_off": [100,300], "dropout": 0.1},
    },
    "A05" : {
        "SNN" : {"learning_rate": 0.01, "weight_decay": 0.0},
        "ANN" : {"learning_rate": 0.001, "weight_decay": 0.5, "cut_off": [75,275], "dropout": 0.1},
    },
    "A06" : {
        "SNN" : {"learning_rate": 0.01, "weight_decay": 0.0},
        "ANN" : {"learning_rate": 0.005, "weight_decay": 0.1, "cut_off": [50,250], "dropout": 0.1},
    },
    "A07" : {
        "SNN" : {"learning_rate": 0.005, "weight_decay": 0.0},
        "ANN" : {"learning_rate": 0.0005, "weight_decay": 0.5, "cut_off": [25,225], "dropout": 0.1},
    },
    "A08" : {
        "SNN" : {"learning_rate": 0.01, "weight_decay": 0.0},
        "ANN" : {"learning_rate": 0.0001, "weight_decay": 0.5, "cut_off": [100,300], "dropout": 0.3},
    },
    "A09" : {
        "SNN" : {"learning_rate": 0.005, "weight_decay": 0.0},
        "ANN" : {"learning_rate": 0.001, "weight_decay": 0.1, "cut_off": [25,225], "dropout": 0.1},
    },
}


asci_models = {}