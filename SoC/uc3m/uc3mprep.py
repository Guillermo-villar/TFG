#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 17:21:12 2023

@author: mlazaro
"""


import logging
import os
import re

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT REQUIRED LIBRARIES
# ----------------------------------------------------------------------------------------------------------------------
# basic python libraries
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# INITIALIZE LOGGER
# ----------------------------------------------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(
    format=FORMAT, level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def func_uniform(df, columns_uni=[], col_label = 'Label', mode='majority', weight_positive=10):
    """
    Applies label processing to pattern that are identical in the specified 
    columns in a DataFrame and adjusts label weights based on the specified mode.

    Parameters:
    + df (pandas.DataFrame): The DataFrame to process.
    + columns_uni (list, optional): List of column names to apply uniform processing to.
       Defaults to an empty list.
    
    + col_label (str, optional): The name of the label column in the DataFrame. 
       Defaults to 'Label'.
    + mode (str, optional): The mode of processing for labels. 
       Supported modes are 'majority', 'single', 'last', and 'weighted_majority' 
       Defaults to 'majority'
    + weight_positive (int, optional): The weight assigned to positive labels. 
        Defaults to 10. Only applied with 'weighted_majority' mode

    Returns:
    pandas.DataFrame: The processed DataFrame with uniform processing applied 
        to the specified columns and adjusted label weights.
    
    Raises:
    ValueError: If an unsupported mode is provided.

    """
    # Group by columns
    # grouped = df.replace('',np.nan).fillna('fillingNaN').groupby(columns_uni)
    # Set the option at the beginning of your script
    pd.set_option('future.no_silent_downcasting', True)
    
    # Your original code remains unchanged
    grouped = df.replace('', np.nan).fillna('fillingNaN').groupby(columns_uni)
        
    # For each group, set 'colX' to the value in the last row of that group
    for name, group in grouped:
        if group.shape[0] > 1:
            if mode == 'last':
                last_value = group.iloc[-1][col_label]
                df.loc[group.index, col_label] = last_value
            elif mode == 'majority':
                maj_value = group[col_label].mean() >= 0.5
                df.loc[group.index, col_label] = maj_value
            elif mode == 'single':
                maj_value = group[col_label].mean() >= 0.0
                df.loc[group.index, col_label] = maj_value
            elif mode == 'weighted_majority':
                threshold = 1/weight_positive
                maj_value = group[col_label].mean() >= threshold
                df.loc[group.index, col_label] = maj_value
            else:
                raise TypeError('Unexpected value for "mode"')
    
    labels = df[col_label]
    logger.info("    Preprocessed alarms: {}".format(df.shape))
    logger.info('      * Labels with distribution : False %d / True %d (IR=%.3f)'%(labels.value_counts()[False], labels.value_counts()[True], labels.value_counts()[False]/labels.value_counts()[True]))
            
    return df
    
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::    


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def func_load_data(
    data_path,
    weight_path,
    alarms_get_labels_colname="reportado",
    alarms_label_colname="Label",
    alarms_timestamp_colname="@timestamp0",
    alarms_time_format="%Y-%m-%d %H:%M:%S",
    feats2drop=[],
):
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    logger.info(" Importing alarms dataframe from: {}".format(data_path))
    df_alarms = pd.read_csv(data_path, low_memory=False)
    # df_alarms = df_alarms.rename(columns={"24x7_8x5": "schedule_L1"})
    df_alarms["schedule_L1"] = df_alarms["24x7_8x5"]  # Create a new column with the desired name
    df_alarms.drop(feats2drop, axis=1, inplace=True)
    df_alarms[alarms_timestamp_colname] = pd.to_datetime(
        df_alarms[alarms_timestamp_colname], format=alarms_time_format
    )
    logger.info("    Alarms loaded : {}".format(df_alarms.shape))

    if weight_path in df_alarms.columns:
        df_alarms["weight"] = df_alarms[weight_path]
    elif os.path.exists(weight_path):
        logger.info(
            " Importing weights dataframe from: {}".format(weight_path)
        )
        df_weights = pd.read_csv(weight_path, low_memory=False)
        logger.info("    Weights loaded : {}".format(df_weights.shape))

        logger.info("  * Adding the weight column")

        dict_weights = {}
        for n_tram in range(df_weights.shape[0]):
            id_tram = df_weights.loc[n_tram, "tramitador"]
            for col in df_weights.columns[1:]:
                dict_weights[(id_tram, col)] = df_weights.loc[n_tram, col]

        df_alarms["weight"] = (
            df_alarms[["tramitador", "proyecto"]]
            .apply(tuple, axis=1)
            .map(dict_weights)
        )

    else:
        raise Exception("Weight path is not a column of dataframe or file")

    # --------------------------------------------------------------------------
    #  Getting Labels
    # --------------------------------------------------------------------------
    logger.info("  * Including labels into the dataframe")

    # Following indications from GMV
    df_alarms.loc[
        df_alarms["motivo_sla_otros"].str.contains("8x5")
        & (df_alarms["proyecto"] != "kring"),
        "escalada_procedimiento",
    ] = True
    if alarms_get_labels_colname == "escalada":
        labels = df_alarms["escalada_procedimiento"] + df_alarms["escalada_L2"]
        # Replace values in 'target' column with False and True
        # df_alarms[alarms_label_colname] = df_alarms[alarms_get_labels_colname].replace({0: False, 1: True})
        df_alarms[alarms_label_colname] = labels.replace({0: False, 1: True})
    elif alarms_get_labels_colname == "tipo_cierre_2":
        df_alarms[alarms_label_colname] = df_alarms[
            alarms_get_labels_colname
        ].replace(
            {
                "confirmed incident": True,
                "no impact": False,
                "false positive": False,
                "potential incident": True,
            }
        )
    elif alarms_get_labels_colname in [
        "reportado",
        "escalada_L2",
        "escalada_procedimiento",
    ]:
        df_alarms[alarms_label_colname] = df_alarms[alarms_get_labels_colname]

    # Getting labels to show statistics
    labels = df_alarms[alarms_label_colname]
    logger.info(
        "  * Included labels with distribution : False %d / True %d (IR=%.3f)"
        % (
            labels.value_counts()[False],
            labels.value_counts()[True],
            labels.value_counts()[False] / labels.value_counts()[True],
        )
    )

    return df_alarms


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def func_preprocessing(
    df_alarms,
    feats2lower=[],
    projects2drop=[],
    drop_titulo=[],
    dict_action_str=None,
    dict_srccountry=None,
    get_top=None,
    project_start_date=None,
    alarms_timestamp_colname="@timestamp0",
    alarms_label_colname="Label",
):
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    logger.info("    Unprocessed alarms : {}".format(df_alarms.shape))
    # -- LOWERCASE
    for col in feats2lower:
        if df_alarms[col].notna().any():  # Check if the column has any non-NaN values
            df_alarms[col] = df_alarms[col].str.lower()

    # -- fuente : removing prefixes
    # df_alarms['fuente'] = df_alarms['fuente'].str.removeprefix('ALERTS:')
    # df_alarms['fuente'] = df_alarms['fuente'].str.removeprefix('ALERTAS:')
    df_alarms.loc[
        df_alarms["fuente"].str.startswith("alertas:", na=False), "fuente"
    ] = df_alarms["fuente"].str.replace("^alertas:", "alerts:", regex=True)
    # Reemplazar valores NaN con una cadena vacía, y a continuación, con 'missing'
    df_alarms["fuente"] = df_alarms["fuente"].fillna("")

    # -- description: remove prefix (contained in use_case_id, e.g. [GMV_GL00013]) and initial whitespaces
    df_alarms["description"] = (
        df_alarms["description"].str.split(pat="]").str[-1].str.strip()
    )

    # -- url_str:
    # Reemplazar valores NaN con una cadena vacía
    df_alarms["url_str"] = df_alarms["url_str"].fillna("")

    # -- user_case_name:
    # Aplicar la función a cada fila
    df_alarms["use_case_in_others"] = df_alarms.apply(
        verificar_contenido, axis=1
    )
    # Reemplazar valores según la condición, manejando cadenas vacías y NaN
    df_alarms.loc[
        (df_alarms["use_case_id"] == "gmv_gl00440")
        & (
            df_alarms["use_case_name"].isna()
            | (df_alarms["use_case_name"] == "")
        ),
        "use_case_name",
    ] = "anomalies-new device on the system"
    df_alarms.loc[
        (df_alarms["use_case_id"] == "hoh_sp00003")
        & (
            df_alarms["use_case_name"].isna()
            | (df_alarms["use_case_name"] == "")
        ),
        "use_case_name",
    ] = "intrusions - ftp anonymous login attempt"
    df_alarms.loc[
        (df_alarms["use_case_id"] == "jac_sp00001")
        & (
            df_alarms["use_case_name"].isna()
            | (df_alarms["use_case_name"] == "")
        ),
        "use_case_name",
    ] = "anomalies - possible compromised account"
    df_alarms.loc[
        (df_alarms["use_case_id"] == "ken_gl00028")
        & (
            df_alarms["use_case_name"].isna()
            | (df_alarms["use_case_name"] == "")
        ),
        "use_case_name",
    ] = "malware - malware detection"
    df_alarms.loc[
        (df_alarms["use_case_id"] == "ken_gl00029")
        & (
            df_alarms["use_case_name"].isna()
            | (df_alarms["use_case_name"] == "")
        ),
        "use_case_name",
    ] = "malware - multiple malware detection same host"
    df_alarms.loc[
        (df_alarms["use_case_id"] == "ono_sp00035")
        & (
            df_alarms["use_case_name"].isna()
            | (df_alarms["use_case_name"] == "")
        ),
        "use_case_name",
    ] = "lucia - incidents report new ticket with investigation required"

    # -- titulo_investigacion: remove prefix (contained in use_case_id, e.g. [GMV_GL00013]) and initial whitespaces
    # df_alarms['titulo_investigacion']=df_alarms['titulo_investigacion'].str.split(pat="]").str[-1].str.strip()

    replacements = {
        "Ã\x83\\u0083Ã\x82Â±": "ñ",
        "ã\x83\\u0083ã\x82â±": "ñ",  # Marce
        "Ã\x83\\u0083Ã\x82Âº": "u",
        "ã\x83\\u0083ã\x82âº": "u",  # Marce
        "Ã\x83\\u0083Ã\x82Â³": "o",
        "ã\x83\\u0083ã\x82â³": "o",  # Marce
        "Ã\x83\\u0083Ã\x82Â¡": "a",
        "ã\x83\\u0083ã\x82â¡": "a",  # Marce
        "Ã\x83\\u0083Ã\x82\\u0083Ã\x83\\u0083Ã\x82\\u0082Ã\x83\\u0082Ã\x82Â¡": "a",
        "ã\x83\\u0083ã\x82\\u0083ã\x83\\u0083ã\x82\\u0082ã\x83\\u0082ã\x82â¡": "a",  # Marce
    }

    df_alarms["titulo_investigacion"] = df_alarms[
        "titulo_investigacion"
    ].apply(lambda x: replace_chars(x, replacements))
    df_alarms = df_alarms.apply(actualizar_campos, axis=1)
    df_alarms["titulo_investigacion"] = df_alarms.apply(clean_title, axis=1)
    # df_alarms = df_alarms[~df_alarms['titulo_investigacion'].str.contains('LUCIA|CERT-EU', regex=True)]

    # -- action_str: translate "auditoría correcta"
    df_alarms.loc[
        df_alarms[
            df_alarms["action_str"].fillna("nada").str.contains("correcta")
        ].index,
        "action_str",
    ] = "audit success"
    df_alarms.loc[
        df_alarms[
            df_alarms["action_str"]
            .fillna("nada")
            .str.lower()
            .str.contains("error de audi")
        ].index,
        "action_str",
    ] = "audit failure"

    if dict_action_str == None:

        dict_action_str = {
            "blocked": "block",
            "dropped": "drop",
            "detected": "detect",
            "detection": "detect",
            "denied": "deny",
            "DENIED": "deny",
            "sucesso da auditoria": "audit success",
            "prevented (blocked)": "prevent",
            "user_login": "login",
            "policyedit": "policy",
            "passthrough": "pass",
            "server-rst": "reset-server",
        }

    for key in dict_action_str.keys():
        df_alarms["action_str"] = df_alarms["action_str"].replace(
            key, dict_action_str[key]
        )  # , inplace=True)
    # Reemplazar los valores 'none' con 'missing'
    df_alarms["action_str"] = df_alarms["action_str"].replace("none", "")
    # Reemplazar las celdas vacías (incluyendo NaN) con 'missing'
    # df_alarms['action_str'] = df_alarms['action_str'].replace('', 'missing')
    df_alarms["action_str"] = df_alarms["action_str"].fillna("")

    # -- srccountry / dstcountry : normalization of values
    if dict_srccountry == None:
        dict_srccountry = {
            "korea, republic of": "korea",
            "united": "united states",
        }

    for key in dict_srccountry.keys():
        df_alarms["srccountry"] = df_alarms["srccountry"].replace(
            key, dict_srccountry[key]
        )  # , inplace=True)
        df_alarms["dstcountry"] = df_alarms["dstcountry"].replace(
            key, dict_srccountry[key]
        )  # , inplace=True)

    df_alarms["srccountry"] = df_alarms["srccountry"].fillna("")
    df_alarms["dstcountry"] = df_alarms["dstcountry"].fillna("")

    # -- user_str
    df_alarms["user_str"] = df_alarms["user_str"].fillna("")

    # -- protocol_str
    df_alarms["protocol_str"] = df_alarms["protocol_str"].fillna("")

    # -- status_str
    df_alarms["status_str"] = df_alarms["status_str"].fillna("")

    # -- id_str
    # Reemplazar los valores 'none' con 'missing'
    df_alarms["id_str"] = df_alarms["id_str"].replace("none", "")
    df_alarms["id_str"] = df_alarms["id_str"].replace("-", "")
    df_alarms["id_str"] = df_alarms["id_str"].fillna("")

    # -- extra_data_str
    dict_extra_data_str = None
    if dict_extra_data_str == None:
        dict_extra_data_str = {"anti-malware": "anti malware"}

    for key in dict_extra_data_str.keys():
        df_alarms["extra_data_str"] = df_alarms["extra_data_str"].replace(
            key, dict_extra_data_str[key]
        )  # , inplace=True)

    # Reemplazar las celdas vacías (incluyendo NaN) con 'missing'
    df_alarms["extra_data_str"] = df_alarms["extra_data_str"].fillna("")

    for text_titulo in drop_titulo:
        df_alarms = df_alarms[
            df_alarms["titulo_investigacion"].str.contains(text_titulo)
            == False
        ]
        # df_alarms = df_alarms[df_alarms['titulo_investigacion'].str.contains('cert-eu')==False]

    for project in projects2drop:
        df_alarms = df_alarms[df_alarms["proyecto"] != project]

    if get_top != None:
        project_count = df_alarms["proyecto"].value_counts()
        top_projects = list(project_count.index)[0:get_top]

        # top_projects = list(project_count.index)[0:get_top]
        top_projects = project_count.index.tolist()[0:get_top]

        list_idx = []

        for project in top_projects:
            list_idx += df_alarms.index[
                df_alarms["proyecto"] == project
            ].tolist()

        list_idx.sort()
        df_alarms = df_alarms.loc[list_idx]

    if project_start_date != None:
        list_idx = []
        for project in project_start_date.keys():
            list_idx += df_alarms.index[
                (df_alarms["proyecto"] == project)
                & (
                    df_alarms[alarms_timestamp_colname]
                    < pd.to_datetime(project_start_date[project])
                )
            ].to_list()

        list_idx.sort()
        df_alarms = df_alarms.drop(list_idx)

    df_alarms.sort_values(by=alarms_timestamp_colname)
    df_alarms.reset_index(inplace=True, drop=True)

    labels = df_alarms[alarms_label_colname]
    logger.info("    Preprocessed alarms: {}".format(df_alarms.shape))
    logger.info(
        "      * Labels with distribution : False %d / True %d (IR=%.3f)"
        % (
            labels.value_counts()[False],
            labels.value_counts()[True],
            labels.value_counts()[False] / labels.value_counts()[True],
        )
    )

    return df_alarms


def replace_chars(text, replacements):
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def clean_title(row):
    title = row["titulo_investigacion"]
    case_id = row["use_case_id"]
    case_name = row["use_case_name"]

    if pd.isna(title) or pd.isna(case_id) or pd.isna(case_name):
        return title

    # Buscar todos los textos entre corchetes y paréntesis
    matches = re.findall(r"\[(.*?)\]", title) + re.findall(r"\((.*?)\)", title)

    for match in matches:
        if match in case_id or match in case_name:
            title = title.replace(f"[{match}]", "").replace(f"({match})", "")

    return title.strip()


# Definir la función para actualizar los campos
def actualizar_campos(row):
    if pd.isna(row["use_case_id"]):
        # Encontrar la cadena dentro de corchetes
        inicio = row["titulo_investigacion"].find("[")
        fin = row["titulo_investigacion"].find("]")
        if inicio != -1 and fin != -1:
            # Obtener la cadena a eliminar y actualizar case_use_id
            cadena_eliminar = row["titulo_investigacion"][inicio : fin + 1]
            # row['use_case_id'] = '[' + cadena_eliminar[1:-1].split(',')[0] + ']'
            row["use_case_id"] = cadena_eliminar[1:-1].split(",")[0]
            # Eliminar la cadena del título de investigación
            row["titulo_investigacion"] = (
                row["titulo_investigacion"]
                .replace(cadena_eliminar, "")
                .strip()
            )
    return row


# Función para verificar si el texto está contenido en las otras dos columnas
def verificar_contenido(row):
    use_case = (
        str(row["use_case_name"]).lower()
        if pd.notnull(row["use_case_name"])
        else ""
    )
    titulo = (
        str(row["titulo_investigacion"]).lower()
        if pd.notnull(row["titulo_investigacion"])
        else ""
    )
    alert = (
        str(row["alert_name"]).lower() if pd.notnull(row["alert_name"]) else ""
    )
    return (use_case in titulo) or (use_case in alert)
