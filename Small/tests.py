#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import csv

import numpy as np
import pandas as pd

# Combine all combinations of dynamic parameters
from itertools import product

from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from uc3m.mlpbayesian import MLPBayesBinW
from uc3m.labelswitching import LSEnsemble


from sklearn.datasets import make_classification


# -----------------------------------------------------------------------------
# INITIALIZE LOGGER
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(
    format=FORMAT, level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

# Initialize best_model_overall
best_model_overall = [None, None, 0, None]

# Initialize best_model_conf
best_model_conf = dict()

# Synthetic Data Generation
n_samples = 6000  # Total samples
n_features = 20    # Features between 30 and 50
n_informative = 15
n_redundant = 5
n_classes = 2
weights = [0.96, 0.04]  # Imbalance ratio IR > 100
random_state = 42

# Generate synthetic data
x, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    n_redundant=n_redundant,
    n_classes=n_classes,
    weights=weights,
    flip_y=0,
    random_state=random_state
)

# Test size
test_size = 0.2

# Train-test split (80-20)
split_idx = int((1-test_size) * n_samples)
x_train, x_test = x[:split_idx], x[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Compute class weights for training data
# class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# cw_train = np.array([class_weights[label] for label in y_train])
cw_train = np.ones(len(y_train))

# Checking data statistics
labels_train = pd.Series(y_train)  # Ensure it's a Series
train_counts = labels_train.value_counts()
print(
    "    + Train labels distribution : False %d / True %d (IR=%.3f)"
    % (
        train_counts[0],
        train_counts[1],
        train_counts[0] / train_counts[1],
    )
)

labels_test = pd.Series(y_test)  # Ensure it's a Series
test_counts = labels_test.value_counts()
print(
    "    + Test labels distribution : False %d / True %d (IR=%.3f)"
    % (
        test_counts[0],
        test_counts[1],
        test_counts[0] / test_counts[1],
    )
)
# Ensure input_size matches data features
input_size = x_train.shape[1]
print(f"Processing data with input size: {input_size}")
print(f"Train data shape: {x_train.shape}, Test data shape: {x_test.shape}")


# Extract input size from x_train (number of features)
input_size = x_train.shape[1]  # Assuming x_train is a 2D array or tensor


print("Processing data ")
print("    + Size of training data : {}".format(x_train.shape))
print("    + Size of test data     : {}".format(x_test.shape))
# Getting labels to show statistics
labels = pd.DataFrame(y_train)
print(
    "    + Train labels distribution : False %d / True %d (IR=%.3f)"
    % (
        labels.value_counts()[0],
        labels.value_counts()[1],
        labels.value_counts()[0] / labels.value_counts()[1],
    )
    )
# Getting labels to show statistics
labels = pd.DataFrame(y_test)
print(
    "    + Test labels distribution : False %d / True %d (IR=%.3f)"
    % (
        labels.value_counts()[0],
        labels.value_counts()[1],
        labels.value_counts()[0] / labels.value_counts()[1],
    )
    )

# ------------------------------------------------------------------------------
# Simulation Params
# ------------------------------------------------------------------------------

# How to select the model: best individiual model ('ind') or best model out of
#                          the best average configuration ('conf')
model_selection = "conf"

n_simus = 3

# Hyperparameters to evaluate

# MLPBayes

# Fixed value arguments
costeN = 1
costeP = 1

s_No = [1]
s_NnBase = [20]
s_pDOent = [0.0]
s_pDOocu = [0.0]
s_tActBase = ["tanh", "relu"]
s_tActSalida = ["identity"]
s_nBatch = [128]
s_nEpoch = [100]


#LSEnsemble
LS_alpha = [0.0] # [0.15, 0.2, 0.25]
LS_beta = [0.0] # [0.0, 0.1]
LS_Q_RB_C = [2.0, 5.0]
LS_Q_RB_S = [1, 2, 5]
LS_num_experts = [21]
LS_hidden_size = [30]
LS_drop_out = [0] #[0 , 0.1]
LS_n_batch = [128]
LS_n_epoch = [50]

# Define model configurations
model_configs = [
    {
        "name": "MLPBayesBinW",
        "class": MLPBayesBinW,
        "params": {
            "class_cost": [costeN, costeP],
            },
        'dynamic_params': list(product(s_No, s_NnBase, s_tActBase, s_tActSalida, s_pDOent, s_pDOocu, s_nBatch, s_nEpoch))
    },
    
    {
        "name": "LSEnsemble",
        "class": LSEnsemble,
        "params": {
            # 'drop_out': 0.0,
            "lbfgs": False, #True, # 
            "mode": "random", # 'class_equitative', #  'representative', # 
            'activation_fn': "relu",
            'loss_fn': 'F1', # 'BCE', # 'KL', #  
            },
        'dynamic_params': list(product(LS_alpha, LS_beta, LS_Q_RB_S, LS_Q_RB_C, 
                                        LS_num_experts, 
                                        LS_hidden_size, LS_drop_out, 
                                        LS_n_batch, LS_n_epoch))
    },
]

model_configs = [model_configs[1]]

# CSV logging setup
csv_file = 'test_results.csv'
csv_columns = ['data_params', 'model_name', 'params', 'metric', 'confusion_matrix']

with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()

    # Data generation parameters
    data_params_list = [
        {'n_samples': 6000, 'n_features': 20, 'n_informative': 15, 'n_redundant': 5, 'n_classes': 2, 'weights': [0.96, 0.04], 'random_state': 42},
        {'n_samples': 8000, 'n_features': 30, 'n_informative': 20, 'n_redundant': 10, 'n_classes': 2, 'weights': [0.95, 0.05], 'random_state': 42},
        # Add more data generation parameter sets as needed
    ]

    for data_params in data_params_list:
        # Generate synthetic data
        x, y = make_classification(
            n_samples=data_params['n_samples'],
            n_features=data_params['n_features'],
            n_informative=data_params['n_informative'],
            n_redundant=data_params['n_redundant'],
            n_classes=data_params['n_classes'],
            weights=data_params['weights'],
            flip_y=0,
            random_state=data_params['random_state']
        )

        # Train-test split (80-20)
        split_idx = int(0.8 * data_params['n_samples'])
        x_train, x_test = x[:split_idx], x[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Compute class weights for training data
        cw_train = np.ones(len(y_train))

        # Ensure input_size matches data features
        input_size = x_train.shape[1]

        # Hyperparameters to evaluate
        # MLPBayes
        s_No = [1]
        s_NnBase = [20]
        s_pDOent = [0.0]
        s_pDOocu = [0.0]
        s_tActBase = ['tanh', 'relu']
        s_tActSalida = ['identity']
        s_nBatch = [128]
        s_nEpoch = [100]

        # LSEnsemble
        LS_alpha = [0.0]
        LS_beta = [0.0]
        LS_Q_RB_C = [2.0, 5.0]
        LS_Q_RB_S = [1, 2, 5]
        LS_num_experts = [21]
        LS_hidden_size = [30]
        LS_drop_out = [0]
        LS_n_batch = [128]
        LS_n_epoch = [50]

        # Define model configurations
        model_configs = [
            {
                'name': 'MLPBayesBinW',
                'class': MLPBayesBinW,
                'params': {
                    'class_cost': [1, 1],
                },
                'dynamic_params': list(product(s_No, s_NnBase, s_tActBase, s_tActSalida, s_pDOent, s_pDOocu, s_nBatch, s_nEpoch))
            },
            {
                'name': 'LSEnsemble',
                'class': LSEnsemble,
                'params': {
                    'lbfgs': False,
                    'mode': 'random',
                    'activation_fn': 'relu',
                    'loss_fn': 'F1',
                },
                'dynamic_params': list(product(LS_alpha, LS_beta, LS_Q_RB_S, LS_Q_RB_C, LS_num_experts, LS_hidden_size, LS_drop_out, LS_n_batch, LS_n_epoch))
            },
        ]

        for model_config in model_configs:
            model_name = model_config['name']
            model_class = model_config['class']
            param_grid = model_config['params']

            CV_config = []
            dynamic_combinations = model_config['dynamic_params']
            n_conf_test = len(dynamic_combinations)
            metric_conf = np.zeros((n_conf_test, 3))

            k_conf = 0
            for dynamic_params in dynamic_combinations:
                k_conf += 1

                if model_name == 'MLPBayesBinW':
                    No, NnBase, tActBase, tActSalida, pDOent, pDOocu, nBatch, nEpoch = dynamic_params
                    nnLayer = (NnBase, )
                    prob_do = [pDOent] + [pDOocu]
                    tAct = [tActBase, tActSalida]
                    config = param_grid.copy()
                    updated_config = config.copy()
                    updated_config.update({
                        'layers_size': nnLayer,
                        'drop_out': prob_do,
                        'activations': tAct,
                        'n_epoch': nEpoch,
                        'n_batch': nBatch,
                    })
                elif model_name == 'LSEnsemble':
                    alpha, beta, Q_RB_S, Q_RB_C, num_experts, hidden_size, drop_out, n_batch, n_epoch = dynamic_params
                    config = param_grid.copy()
                    updated_config = config.copy()
                    updated_config.update({
                        'alpha': alpha,
                        'beta': beta,
                        'Q_RB_S': Q_RB_S,
                        'Q_RB_C': Q_RB_C,
                        'num_experts': num_experts,
                        'hidden_size': hidden_size,
                        'drop_out': drop_out,
                        'n_batch': n_batch,
                        'n_epoch': n_epoch,
                        'input_size': input_size
                    })

                CV_config.append(updated_config)

            k_conf = 0
            for cv_config in CV_config:
                for k_simu in range(3):
                    model = model_class(**cv_config)
                    model.fit(x_train, y_train, sample_weight=cw_train)
                    ye_test = model.predict(x_test)
                    CM = confusion_matrix(y_test, ye_test)
                    metric = balanced_accuracy_score(y_test, ye_test)
                    metric_conf[k_conf, k_simu] = metric

                    # Update best_model_overall during model evaluation
                    if metric > best_model_overall[2]:
                        best_model_overall = [model, cv_config, metric, CM]
                        logger.info(f'Best configuration overall: {cv_config}')
                        logger.info(f'Best metric overall: {best_model_overall[2]:.5f}')

                    writer.writerow({
                        'data_params': data_params,
                        'model_name': model_name,
                        'params': cv_config,
                        'metric': metric,
                        'confusion_matrix': CM.tolist()
                    })

                k_conf += 1

# Report best model and performance
logger.info(f"Best overall model: {best_model_overall[0].__class__.__name__}")

for model_config in model_configs:
    model_name = model_config["name"]
    logger.info(f"Model: {model_name}")
    logger.info("-----------------------------------------------------")
    logger.info('Peak Performance')
    logger.info(f"Selected configuration {best_model_overall[1]}")
    logger.info('Best Peak performance: BAcc = {:.5f}'.format(best_model_overall[2]))
    logger.info("-----------------------------------------------------")
    
    idx_best = np.argmax(np.mean(metric_conf[model_name], axis=1))
    model_conf = best_model_conf[model_name][idx_best]
    logger.info('Average Performance')
    logger.info(f"Selected configuration {model_conf[1]}")
    logger.info('Best Average performance: BAcc = {:.5f}'.format(model_conf[2]))

if model_selection == "ind":
    saved_model = best_model_overall

elif model_selection == "conf":
    for model_config in model_configs:
        model_name = model_config["name"]
        idx_best = np.argmax(np.mean(metric_conf[model_name], axis=1))
        saved_model = best_model_conf[model_name][idx_best]
        
else:
    raise TypeError('Unexpected value for "model_selection"')
