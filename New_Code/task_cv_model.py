#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:52:14 2024

@author: mlazaro
"""

import datetime
import json
import logging
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Combine all combinations of dynamic parameters
from itertools import product


from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from libraries.mlpbayesian import MLPBayesBinW
from libraries.labelswitching import LSEnsemble

# -----------------------------------------------------------------------------
# IMPORT REQUIRED LIBRARIES
# -----------------------------------------------------------------------------
# basic python libraries


plt.rcdefaults()
plt.rcParams.update(
    {"font.size": 14}
)  # You can adjust the font size as needed

# root_path = '/Users/fran/fran@ing.uc3m.es - Google Drive/Mi unidad'
root_path = '/Volumes/Lacie_8Tb/Users/fran/Machine_Learning'
# root_path = '/Users/fran/Machine_Learning'


working_root_path = 'SmartNOC/alert_generation_def'
# working_root_path = 'SmartNOC'


main_path = os.path.join(root_path,working_root_path)

# Scenarios: Baseline_Health, Degraded_Health, Incident_Health, SmartNOC

split_idx_path = 'data/interim/coding_data/Dataminer/Full/SmartNOC/split_idx_train_test_all_column_5Fold_no_context_CNNmulti512.json'
coded_data_path = 'data/interim/coding_data/Dataminer/Full/SmartNOC/train_test_all_column_5Fold_no_context_CNNmulti512.pkl'
predict_path = 'predict_data/Dataminer/Full/SmartNOC/no_context'
model_path = 'models/SmartNOC/no_context'


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


# ------------------------------------------------------------------------------
# Input / Output files
# ------------------------------------------------------------------------------


if not os.path.exists(os.path.join(main_path, predict_path)):
    logger.info(" New folder for prediction results created: {}".format(predict_path))
    os.makedirs(os.path.join(main_path, predict_path))

sess_date = '' # datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_file = "model_" + sess_date
model_save_path = os.path.join(model_path, model_file)
if not os.path.exists(os.path.join(main_path, model_save_path)):
    logger.info(" New folder for model saving created: {}".format(model_save_path))
    os.makedirs(os.path.join(main_path, model_save_path))    

#--------------------------------------------------------------------------
# Loading input data
#--------------------------------------------------------------------------        
if os.path.exists(os.path.join(main_path, coded_data_path)):
    logger.info('Loading coded data from file {}'.format(os.path.join(main_path, coded_data_path)))
    x_train_list, x_test_list, y_train_list, y_test_list, coding_labels, nlp_feats, df_train_list, df_test_list = pd.read_pickle(coded_data_path)[:]
    # x_train_list, x_test_list, y_train_list, y_test_list, coding_labels, nlp_feats, df_alarms = pd.read_pickle(coded_data_path)[:]
    if isinstance(x_train_list, list):
        num_folds = len(x_train_list)
    else:   # num_folds = 1
        num_folds = 1
        x_train_list = [x_train_list]
        x_test_list = [x_test_list]
        y_train_list = [y_train_list]
        y_test_list = [y_test_list]
        df_train_list = [df_train_list]
        df_test_list = [df_test_list]
    df_alarms = pd.concat([df_train_list[0], df_test_list[0]]).sort_index()
    
    # Extract input size from x_train (number of features)
    input_size = x_train_list[0].shape[1]  # Assuming x_train is a 2D array or tensor
else:
    raise NameError('Coded data file not found: {}'.format(coded_data_path))
    
if os.path.exists(os.path.join(main_path, split_idx_path)):        
    logger.info('Loading N-fold indexes from file {}'.format(os.path.join(main_path, split_idx_path)))
    with open(os.path.join(main_path, split_idx_path), 'r') as j:
        split_idx_dict = json.loads(j.read())
else:
    raise NameError('N-fold indexes file not found: {}'.format(coded_data_path))
        

# ------------------------------------------------------------------------------
# Simulation Params
# ------------------------------------------------------------------------------
# How to select the model: best individiual model ('ind') or best model out of
#                          the best average configuration ('conf')
model_selection = "conf"



###############################################################################
###                          MAIN EXECUTION                                 ###
###############################################################################


# Train and evaluation of the model specified above in 'classifiers'
print("MODEL SELECTION AND TRAINING: Starting model evaluation")


nSamples = 10000
# num_folds = 2
x_train_list = x_train_list[:num_folds]
x_test_list = x_test_list[:num_folds]
y_train_list = y_train_list[:num_folds]
y_test_list = y_test_list[:num_folds]

n_simus = 3

# Hyperparameters to evaluate

# MLPBayes

# Fixed value arguments
costeN = 1
costeP = 1

s_No = [1]
s_NnBase = [20, 30, 50, 60]
s_pDOent = [0.0]
s_pDOocu = [0.0, 0.1, 0.2]
s_tActBase = ["tanh", "relu"]
s_tActSalida = ["identity"]
s_nBatch = [128, 256]
s_nEpoch = [100, 200]


#LSEnsemble
LS_alpha = [0.43, 0.45]
LS_beta = [0.0, 0.05]
LS_Q_RB_C = [1.0, 2.0]
LS_Q_RB_S = [70, 100]
LS_num_experts = [31]
LS_hidden_size = [30]
LS_n_batch = [128]
LS_n_epoch = [100]

#LSEnsemble
LS_alpha = [0.25, 0.3, 0.35, 0.4]
LS_beta = [0.0, 0.05]
LS_Q_RB_C = [1.0]
LS_Q_RB_S = [1,  10]
LS_num_experts = [31]
LS_hidden_size = [30]
LS_drop_out = [0.0]
LS_n_batch = [128]
LS_n_epoch = [100]

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
             # 'drop_out': 0,
             "lbfgs": False, #True, # 
             "mode": 'representative', # "random", # 'class_equitative' #  
             "activation_fn": "relu",
             "loss_fn": "F1", # "KL", # "MSE", #
             },
         'dynamic_params': list(product(LS_alpha, LS_beta, LS_Q_RB_S, LS_Q_RB_C, 
                                       LS_num_experts, 
                                       LS_hidden_size, LS_drop_out, LS_n_batch, 
                                       LS_n_epoch))
    },
]

model_configs = [model_configs[1]] # Just LSEnsemble

# Main Loop
logger.info("Training ML models (this can take some time)...")

best_metric_overall = 0
best_model_overall = None

metric_conf = dict()
best_metric_conf = dict()
best_model_conf = dict()

for model_config in model_configs:
    model_name = model_config["name"]
    model_class = model_config["class"]
    param_grid = model_config["params"]
    
    logger.info(f"Training and evaluating model: {model_name}")

    CV_config = []
    dynamic_combinations = model_config["dynamic_params"]
    n_conf_test = len(dynamic_combinations)
    best_model_conf[model_name] = [dict() for _ in range(n_conf_test)]
    metric_conf[model_name] = np.zeros((n_conf_test, num_folds, n_simus))
   
    k_conf = 0
    # Iterate over dynamic parameter combinations
    for dynamic_params in dynamic_combinations:
        k_conf += 1
        logger.info(f" * Configuration {k_conf}/{len(dynamic_combinations)} for {model_name}")

        # Update dynamic parameters for MLPBayesBinW
        if model_name == "MLPBayesBinW":
            No, NnBase, tActBase, tActSalida, pDOent, pDOocu, nBatch, nEpoch = dynamic_params
            
            # Generate the corresponding dynamic parameters
            nnLayer = (NnBase, ) # tuple([s_NnBase[0] for _ in range(s_No[0])])  # Example: fixed layer sizes
            prob_do = [pDOent] + [pDOocu]
            tAct = [tActBase, tActSalida]
            
            # Create a base config from the model's param_grid
            config = param_grid.copy()  # Start with the base configuration
    
            # Update the config with dynamic values
            updated_config = config.copy()
            updated_config.update({
                "layers_size": nnLayer,
                "drop_out": prob_do,
                "activations": tAct,
                'n_epoch': nEpoch,
                'n_batch': nBatch,
            })
            
        elif model_name == "LSEnsemble":
            alpha, beta, Q_RB_S, Q_RB_C, num_experts, hidden_size, drop_out, n_batch, n_epoch = dynamic_params
            # Create a base config from the model's param_grid
            config = param_grid.copy()  # Start with the base configuration
    
            # Update the config with dynamic values
            updated_config = config.copy()
            updated_config.update({
                "alpha": alpha,
                "beta": beta,
                "Q_RB_S": Q_RB_S,
                "Q_RB_C": Q_RB_C,
                'num_experts': num_experts,
                'hidden_size': hidden_size,
                'drop_out': drop_out,
                'n_batch': n_batch,
                'n_epoch': n_epoch,
                'input_size': input_size
            })
        
        # Append the updated config to the output list
        CV_config.append(updated_config)
    
    # Now proceed with the model training and evaluation logic     
    best_metric_conf[model_name] = dict()
    k_conf = 0
    for cv_config in CV_config:
        best_metric_conf[model_name] = [[] for _ in range(num_folds)]
        best_model_conf[model_name][k_conf] = [[] for _ in range(num_folds)]
        for nFold in range(num_folds):        
            
            logger.info("  * Processing Fold {}".format(nFold+1))
            
            # Get the total number of samples in the training set
            num_total_samples = x_train_list[nFold].shape[0]
            
            # Randomly select indices without replacement
            random_indices = np.random.choice(num_total_samples, nSamples, replace=False)
            
            # Select training samples using the random indices
            x_train = x_train_list[nFold][random_indices]
            y_train = y_train_list[nFold][random_indices]
            # Keep the test set unchanged
            x_test = x_test_list[nFold]
            y_test = y_test_list[nFold]
    
            cw_train = np.ones(len(y_train))
    
            
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
    
            best_metric_conf[model_name][nFold] = 0
            for k_simu in range(n_simus):
                logger.info(f"    + Realization {k_simu+1}/{n_simus} for configuration {k_conf+1}/{len(dynamic_combinations)}")
                
                model = model_class(**cv_config)
                
                # Train the model
                model.fit(x_train, y_train, sample_weight=cw_train)
                
                # Evaluate the model
                ye_test = model.predict(x_test)
                CM = confusion_matrix(y_test, ye_test)
                metric = balanced_accuracy_score(y_test, ye_test)
                metric_conf[model_name][k_conf, nFold, k_simu] = metric
                
                if metric > best_metric_overall:
                    best_metric_overall = metric
                    best_model_overall = [model, cv_config, metric, CM]
                    logger.info(f'Best configuration overall: {cv_config}')
                    logger.info(f'Best metric overall: {best_metric_overall:.5f}')
                    
            
            # Compute the averaged metric for this configuration after all simulations
            avg_metric_conf = np.mean(metric_conf[model_name][k_conf, nFold,:])
            
            # Update best metric and model for the current configuration
            if avg_metric_conf > best_metric_conf[model_name][nFold]:
                best_metric_conf[model_name][nFold] = avg_metric_conf
                best_model_conf[model_name][k_conf][nFold] = [model, cv_config, avg_metric_conf, CM]
                logger.info(f'Best configuration conf: {cv_config}')
                logger.info(f'Best metric conf: {avg_metric_conf:.5f}')
                logger.info(f'Best CM conf: {CM}')
            
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
    model_conf = best_model_conf[model_name][idx_best][0]
    logger.info('Average Performance')
    logger.info(f"Selected configuration {model_conf[1]}")
    logger.info('Best Average performance: BAcc = {:.5f}'.format(model_conf[2]))

if model_selection == "ind":
    saved_model = best_model_overall

    with open(model_save_path + ".joblib", "wb") as f:  # Python 3: open(..., 'wb')
        joblib.dump(saved_model, f)

    logger.info("Model saved in {}".format(model_save_path))

elif model_selection == "conf":
    for model_config in model_configs:
        model_name = model_config["name"]
        idx_best = np.argmax(np.mean(metric_conf[model_name], axis=1))
        saved_model = best_model_conf[model_name][idx_best][0]
        
        with open(model_save_path + '_'+model_name+".joblib", "wb") as f:  # Python 3: open(..., 'wb')
            joblib.dump(saved_model, f)

        logger.info("Model saved in {}".format(model_save_path))

else:
    raise TypeError('Unexpected value for "model_selection"')

