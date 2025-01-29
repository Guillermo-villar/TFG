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
import yaml
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
data_path = os.path.join(FILLED_DATA, "dataset_filled.csv")
weight_path = "sample_weight_dinamic_expertise"

split_path = "split_idx"
coding_path = "encoding"
output_path = "output"

split_idx_file = "split_idx_train_test_ex_uni.json"
data_cod_file = "coded_data_esc_june_ex_uni.pkl"
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
reset_all = False # True
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
print("LOADING : Alarms from {}".format(data_path))
print("          Weights from {}".format(weight_path))
df_alarms = func_load_data(
    data_path,
    weight_path,
    alarms_get_labels_colname=alarms_get_labels_colname,
    feats2drop=feats2drop,
)

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

# Hyperparameters to evaluate
s_No = [1]
s_NnBase = [10, 30, 50]
s_pDOent = [0.0]
s_pDOocu = [0.0, 0.1]
s_tActBase = ["tanh", "relu"]
s_tActSalida = ["identity"]
s_nBatch = [128, 256]
s_nEpoch = [10, 25, 50, 100]

# Fixed value arguments
costeN = 1
costeP = 1
n_simus = 3

n_conf_test = len(s_No) * len(s_NnBase) * len(s_pDOent) * len(s_pDOocu)
n_conf_test *= len(s_tActBase) * len(s_tActSalida)
n_conf_test *= len(s_nBatch) * len(s_nEpoch)
n_models_test = n_conf_test * n_simus

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
x_train = x_train_list[nFold]
x_test = x_test_list[nFold]
y_train = y_train_list[nFold]
y_test = y_test_list[nFold]

cw_train = df_train_list[nFold]["weight"].to_numpy()

print("Processing data ")
print("    + Size of training data : {}".format(x_train.shape))
print("    + Size of test data     : {}".format(x_test.shape))

k_model = 0
k_conf = 0
best_metric = 0
best_model_conf = [[] for x in range(n_conf_test)]
metric_conf = np.zeros((n_conf_test, n_simus))
print("Training MLP models (this can take some time)....")
for No in s_No:
    for NnBase in s_NnBase:
        for pDOent in s_pDOent:
            for pDOocu in s_pDOocu:
                for tActBase in s_tActBase:
                    for tActSalida in s_tActSalida:
                        nnLayer = tuple([NnBase for k in range(No)])
                        prob_do = [pDOent] + [pDOocu for k in range(No)]
                        tAct = [tActBase for k in range(No)] + [tActSalida]
                        for nBatch in s_nBatch:
                            for nEpoch in s_nEpoch:
                                best_metric_conf = 0
                                print(
                                    " * Configuration {} of {}".format(
                                        k_conf + 1, n_conf_test
                                    )
                                )
                                for k_simu in range(n_simus):
                                    k_model += 1
                                    print(
                                        "    + Realization {} of {} (MLP model {} of {}) ....".format(
                                            k_simu + 1,
                                            n_simus,
                                            k_model,
                                            n_models_test,
                                        )
                                    )
                                    model = MLPBayesBinW(
                                        n_epoch=nEpoch,
                                        n_batch=nBatch,
                                        layers_size=nnLayer,
                                        activations=tAct,
                                        drop_out=prob_do,
                                        class_cost=[costeN, costeP],
                                    )

                                    model.fit(
                                        x_train,
                                        y_train,
                                        sample_weight=cw_train,
                                    )
                                    ye_test = model.predict(x_test)

                                    CM = confusion_matrix(y_test, ye_test)
                                    metric = balanced_accuracy_score(
                                        y_test, ye_test
                                    )
                                    metric_conf[k_conf, k_simu] = metric

                                    if metric > best_metric_conf:
                                        best_metric_conf = metric
                                        best_model_conf[k_conf] = [
                                            model,
                                            best_metric_conf,
                                            CM,
                                        ]

                                    if metric > best_metric:
                                        best_metric = metric
                                        best_model = [model, best_metric, CM]

                                k_conf += 1

if model_selection == "ind":
    saved_model = best_model
elif model_selection == "conf":
    idx_best = np.argmax(np.mean(metric_conf, axis=1))
    saved_model = best_model_conf[idx_best]
else:
    raise TypeError('Unexpected value for "model_selection"')

print("Selected model performance: {} BAcc".format(saved_model[1]))

with open(model_save_path + ".joblib", "wb") as f:  # Python 3: open(..., 'wb')
    joblib.dump(saved_model, f)

print("Model saved in {}".format(model_save_path))
