#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:52:14 2024

@author: fran
"""

import datetime
import json
import logging
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Combine all combinations of dynamic parameters
from itertools import product


from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from uc3m.mlpbayesian import MLPBayesBinW
from uc3m.labelswitching import LSEnsemble
from uc3m.uc3mcod import func_encoding2, func_idx_train_test
from uc3m.uc3mprep import func_load_data, func_preprocessing

# -----------------------------------------------------------------------------
# IMPORT REQUIRED LIBRARIES
# -----------------------------------------------------------------------------
# basic python libraries


plt.rcdefaults()
plt.rcParams.update(
    {"font.size": 14}
)  # You can adjust the font size as needed

FILLED_DATA = 'data/interim/preprocess_v0/preprocess_2024/dataset_filled/'
MODEL_PATH = 'models/uc3m_model/'

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


sess_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#sess_date = "20240723-080558" # Custom subversion TODO: Tener cuidado con esto
with open("config/inference_config_plantilla.yaml", "r") as file:
    original_dict = yaml.safe_load(file)

original_dict["params"]["subversion"] = sess_date
original_dict["params"]["verbose"] = True # Custom verbose a True

with open("config/inference_config_" + sess_date + ".yaml", "w") as file:
    yaml.dump(original_dict, file, default_flow_style=False)


# ------------------------------------------------------------------------------
# Input / Output files
# ------------------------------------------------------------------------------
data_path = [os.path.join(FILLED_DATA, "dataset_filled.csv")]
# data_path.append('data/interim/raw/dataset_2024_07.csv')  # Example additional dataset
# data_path.append('data/interim/rawdataset_full_2023_2024.csv')  # Example additional dataset
data_path = ['data/interim/raw/dataset_full_2023_2024.csv']

test_data_path = None # 'data/interim/raw/dataset_2024_09.csv'

# Initialize an empty list for data paths to process
valid_data_paths = []

# Validate each data path
for path in data_path:
    if os.path.exists(path):
        print(f"LOADING: File found - {path}")
        valid_data_paths.append(path)
    else:
        print(f"WARNING: File not found - {path}")

# Ensure at least one file exists
if not valid_data_paths:
    raise FileNotFoundError("No valid data files found in the specified paths.")

weight_path = "sample_weight_dinamic_expertise"

split_path = "split_idx"
coding_path = "encoding"
output_path = "output"

split_idx_file = "split_idx_train_test_ex_uni.json"
data_cod_file = "coded_data_esc_dec_ex_uni.pkl"
model_file = "model_" + sess_date


if not os.path.exists(os.path.join(MODEL_PATH, coding_path)):
    print(" New folder created: {}".format(coding_path))
    os.makedirs(os.path.join(MODEL_PATH, coding_path))

if not os.path.exists(os.path.join(MODEL_PATH, split_path)):
    print(" New folder created: {}".format(split_path))
    os.makedirs(os.path.join(MODEL_PATH, split_path))

if not os.path.exists(os.path.join(MODEL_PATH, output_path)):
    print(" New folder created: {}".format(output_path))
    os.makedirs(os.path.join(MODEL_PATH, output_path))


# ------------------------------------------------------------------------------
# Simulation Params
# ------------------------------------------------------------------------------
# Reset or not all previous encoded data files if they already exist
reset_all = False # True # 
# How to select the model: best individiual model ('ind') or best model out of
#                          the best average configuration ('conf')
model_selection = "conf"
# Test size
test_size = 0.1

# Preprocessing parameters
alarms_label_colname = "Label"
alarms_get_labels_colname = (
    "escalada"  # 'escalada', 'reportado', 'tipo_cierre_2'
)

drop_titulo = ["lucia", "cert-eu"]
use_NLP = True
encoding_NLP = ["CNNmulti512", "SWIVEL20"]
split_idx_type = "sequential"
projects_drop = [
    "insta",
    "schlutia",
    "onodajunjiro",
    "savonlinna",
]  # ,'kring']

alarms_timestamp_colname = "@timestamp0"
alarms_time_format = "%Y-%m-%d %H:%M:%S"

get_top = None
project_start_date = None

# Type of binary encoding [string] 'OneZero', 'PlusMinusOne'
binary_type = "PlusMinusOne"
bincat_dict = {"schedule_L1": "24x7"}
# Type of normalization of muneric data [string, 'MinMax', 'MinMaxOne, 'Standard', 'MaxAbs']
normalization = "Standard"
# ------------------------------------------------------------------------------
# DEFINING FEATURES
# ------------------------------------------------------------------------------
# Define the features for different categories
nlp_feats = [
    ["titulo_investigacion", "use_case_name", "url_str", "id_str"],
    ["extra_data_str", "action_str", "fuente"],
    ["alert_name"],
]
"""
purely_categorical_feats = [
    "proyecto",
    "severidad",
    "srccountry",
    "dstcountry",
    "user_str",
    "protocol_str",
    "status_str",
]
"""
purely_categorical_feats = [
    "proyecto",
    "severidad",
    "srccountry",
    "dstcountry",
    "user_str",
    "protocol_str",
    "status_str",
    "use_case_id"
]
multiple_categorical_feats = []
numeric_feats = ["AlertLevel"]
#numeric_feats = ["AlertLevel", "numero_eventos"]
binary_feats = [
    "schedule_L1",
    "srcip_exists",
    "dstip_exists",
    "srcip_public",
    "dstip_public",
]

if len(nlp_feats) > 0:
    if type(nlp_feats[0]) == str:
        feats2lower = nlp_feats + purely_categorical_feats + ["description"]
    else:
        feats2lower = (
            sum(nlp_feats, []) + purely_categorical_feats + ["description"]
        )
else:
    feats2lower = purely_categorical_feats + ["description"]


# --- GROUPING ALL FEATURES ---------------------------------------------------
features = [
    binary_feats,
    numeric_feats,
    purely_categorical_feats,
    multiple_categorical_feats,
    nlp_feats,
]

feats2drop = [
    "id_nuntiare",
    "duplicada",
    "motivo_sla",
]

feats2keep = [
    "@timestamp0",
    "id_alerta",
    "titulo_investigacion",
    "numero_eventos",
    "proyecto",
    "schedule_L1",
    "severidad",
    "alert_name",
    "AlertLevel",
    "escalada_L2",
    "escalada_procedimiento",
    "reportado",
    "fuente",
    "weight",
    "Label",
]

###############################################################################
###                          MAIN EXECUTION                                 ###
###############################################################################

# Loading alarms data
print(f"LOADING: Alarms from {valid_data_paths}")
print("          Weights from {}".format(weight_path))
df_alarms_list = [
    func_load_data(
        path,
        weight_path,
        alarms_get_labels_colname=alarms_get_labels_colname,
        feats2drop=feats2drop,
    )
    for path in valid_data_paths
]

# Concatenate and sort by date
df_alarms = pd.concat(df_alarms_list, ignore_index=True)
df_alarms = df_alarms.sort_values(by="@timestamp0").reset_index(drop=True)

nSamples = 10000
# df_alarms = df_alarms[-nSamples:]


if (test_size == 0) and (test_data_path is not None):
    test_data = True
    # Loading test alarms data
    logger.info('LOADING : Test Alarms from {}'.format(test_data_path))
    
    df_test_alarms = func_load_data(test_data_path, weight_path, 
                                    alarms_get_labels_colname = alarms_get_labels_colname,
                                    feats2drop=feats2drop
                                    )
else:
    test_data = False
    

# Preprocessing alarms data
print("PREPROCESSING Alarms")
df_alarms = func_preprocessing(
    df_alarms,
    drop_titulo=drop_titulo,
    projects2drop=projects_drop,
    get_top=get_top,
    project_start_date=project_start_date,
    feats2lower=feats2lower,
)

if test_data:
    # Preprocessing test alarms data
    logger.info('PREPROCESSING Test Alarms')
   
    df_test_alarms = func_preprocessing(df_test_alarms, drop_titulo=drop_titulo,
                                       projects2drop=projects_drop,
                                       get_top=get_top,
                                       project_start_date=project_start_date,
                                       feats2lower=feats2lower)
    
# Train and test split
split_idx_path = os.path.join(MODEL_PATH, split_path, split_idx_file)
if (os.path.exists(split_idx_path) == False) or reset_all:
    print("TASK : Train / Test split : {}".format(split_idx_type))
    split_idx_dict = func_idx_train_test(
        df_alarms[alarms_label_colname],
        split_idx_path=split_idx_path,
        split_idx_type=split_idx_type,
        test_size=test_size,
    )
else:
    print("LOADING : Train / Test split from {}".format(split_idx_path))
    with open(split_idx_path, "r") as j:
        split_idx_dict = json.loads(j.read())

    if np.max(split_idx_dict["train0"] + split_idx_dict["test0"]) != (
        df_alarms.shape[0] - 1
    ):
        raise ValueError(
            "Indexes of folds are not compatible with dataframe size..."
        )

# Encoding of the different features (conversion to numerical values)
data_cod_path = os.path.join(MODEL_PATH, coding_path, data_cod_file)
if (os.path.exists(data_cod_path) == False) or reset_all:
    print("TASK : Encoding")
    coded_data = func_encoding2(
        df_alarms,
        features,
        bincat_dict,
        split_idx=split_idx_dict,
        binary_type=binary_type,
        normalization=normalization,
        num_folds=1,
        text_enc=encoding_NLP,
        feats2keep=feats2keep,
        data_out_path=data_cod_path,
        model_path=original_dict["deps"]["model"]["encoder_path"],
        subversion=sess_date,
    )

else:
    print("LOADING : Encoding from {}".format(data_cod_path))
    coded_data = pd.read_pickle(data_cod_path)[:]

# Train and evaluation of the model specified above in 'classifiers'
print("MODEL SELECTION AND TRAINING: Starting model evaluation")


model_save_path = os.path.join(MODEL_PATH, model_file)


(
    x_train_list,
    x_test_list,
    y_train_list,
    y_test_list,
    idx_test_list,
    coding_labels,
    nlp_feats,
    df_train_list,
    df_test_list,
    binary_type,
    normalization,
    text_enc,
) = coded_data



nFold = 0
x_train = x_train_list[nFold][-nSamples:]
x_test = x_test_list[nFold]# [-5000:]
y_train = y_train_list[nFold][-nSamples:]
y_test = y_test_list[nFold]# [-5000:]

cw_train = df_train_list[nFold]["weight"].to_numpy()[-nSamples:]

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
LS_alpha = [0.43, 0.46]
LS_beta = [0.0]
LS_Q_RB_C = [1.0]
LS_Q_RB_S = [50, 60]
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
    best_model_conf[model_name] = [[] for _ in range(n_conf_test)]
    metric_conf[model_name] = np.zeros((n_conf_test, n_simus))
   
    
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
    best_metric_conf[model_name] = 0
    k_conf = 0 
    for cv_config in CV_config:
        for k_simu in range(n_simus):
            logger.info(f"    + Realization {k_simu+1}/{n_simus} for configuration {k_conf+1}/{len(dynamic_combinations)}")
            
            model = model_class(**cv_config)
            
            # Train the model
            model.fit(x_train, y_train, sample_weight=cw_train)
            
            # Evaluate the model
            ye_test = model.predict(x_test)
            CM = confusion_matrix(y_test, ye_test)
            metric = balanced_accuracy_score(y_test, ye_test)
            metric_conf[model_name][k_conf, k_simu] = metric
            
            if metric > best_metric_overall:
                best_metric_overall = metric
                best_model_overall = [model, cv_config, metric, CM]
                logger.info(f'Best configuration overall: {cv_config}')
                logger.info(f'Best metric overall: {best_metric_overall:.5f}')
                
        
        # Compute the averaged metric for this configuration after all simulations
        avg_metric_conf = np.mean(metric_conf[model_name][k_conf, :])
        
        # Update best metric and model for the current configuration
        if avg_metric_conf > best_metric_conf[model_name]:
            best_metric_conf[model_name] = avg_metric_conf
            best_model_conf[model_name][k_conf] = [model, cv_config, avg_metric_conf, CM]
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
    model_conf = best_model_conf[model_name][idx_best]
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
        saved_model = best_model_conf[model_name][idx_best]
        
        with open(model_save_path + '_'+model_name+".joblib", "wb") as f:  # Python 3: open(..., 'wb')
            joblib.dump(saved_model, f)

        logger.info("Model saved in {}".format(model_save_path))

else:
    raise TypeError('Unexpected value for "model_selection"')

