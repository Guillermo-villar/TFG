#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import csv
import time
import os
import numpy as np
import pandas as pd
import signal
import json
# Combine all combinations of dynamic parameters
from itertools import product
from sklearn.metrics import (balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix)
from uc3m.mlpbayesian import MLPBayesBinW
from uc3m.labelswitching import LSEnsemble

# -----------------------------------------------------------------------------
# INITIALIZE LOGGER
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

def setup_logging(log_level="INFO"):
    """Set up logging with the specified level"""
    logging.getLogger().setLevel(getattr(logging, log_level))
    FORMAT = "%(asctime)-15s %(message)s"
    logging.basicConfig(
        format=FORMAT, level=getattr(logging, log_level), datefmt="%Y-%m-%d %H:%M:%S"
    )
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)

def custom_metric(y_true, y_pred):
    """
    Returns:
        metric: 0.5 * balanced_accuracy + 0.5 * (1 - minority error)
        balanced_accuracy: sklearn's balanced_accuracy_score
        minority_score: 1 - minority error
    """
    bac = balanced_accuracy_score(y_true, y_pred)
    unique, counts = np.unique(y_true, return_counts=True)
    if len(unique) < 2:
        minority_error = 0.0
    else:
        minority_class = unique[np.argmin(counts)]
        minority_mask = (y_true == minority_class)
        if minority_mask.sum() == 0:
            minority_error = 0.0
        else:
            minority_error = np.mean(y_pred[minority_mask] != y_true[minority_mask])
    minority_score = 1 - minority_error
    metric = 0.5 * bac + 0.5 * minority_score
    return metric, bac, minority_score

def run_test_from_csv(
    dataset_path, test_size, model_config, output_config, dataset_params=None, data_params_list=None,
    use_previous_best_runner=False, best_runner_file=None, study_mode="full_study"
):
    """
    Run the ensemble test with data loaded from a single CSV file or multiple data parameter sets.
    If data_params_list is provided, loop over each set and run the workflow for each.
    """
    setup_logging(output_config.get("log_level", "INFO"))
    csv_file = output_config.get("csv_file", "test_results.csv")
    os.makedirs(os.path.dirname(csv_file) if os.path.dirname(csv_file) else '.', exist_ok=True)
    csv_columns = [
        'data_params', 'model_name', 'params', 'metric', 'confusion_matrix', 'accuracy', 'time_taken',
        'label_switching_used', 'num_labels_switched_pos_to_neg', 'total_pos_labels', 'pct_pos_switched',
        'num_labels_switched_neg_to_pos', 'total_neg_labels', 'pct_neg_switched',
        'custom_metric', 'balanced_accuracy', 'minority_score'
    ]

    # Open with immediate flush mode
    with open(csv_file, 'w', newline='', buffering=1) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        csvfile.flush()
        os.fsync(csvfile.fileno())  # Force writing to disk

        # If data_params_list is provided, loop over it, else just run once
        if data_params_list is not None:
            for data_params in data_params_list:
                # Load dataset from CSV (user must provide correct CSV for each data_params)
                logger.info(f"Loading dataset from {dataset_path} for data_params: {data_params}")
                df = pd.read_csv(dataset_path)
                # ... Optionally filter/modify df based on data_params if needed ...
                _run_single_experiment(df, test_size, model_config, output_config, data_stats_extra=data_params, writer=writer, csvfile=csvfile,
                                       use_previous_best_runner=use_previous_best_runner, best_runner_file=best_runner_file, study_mode=study_mode)
        else:
            logger.info(f"Loading dataset from {dataset_path}")
            df = pd.read_csv(dataset_path)
            _run_single_experiment(df, test_size, model_config, output_config, data_stats_extra=dataset_params, writer=writer, csvfile=csvfile,
                                   use_previous_best_runner=use_previous_best_runner, best_runner_file=best_runner_file, study_mode=study_mode)

def _run_single_experiment(df, test_size, model_config, output_config, data_stats_extra=None, writer=None, csvfile=None,
                           use_previous_best_runner=False, best_runner_file=None, study_mode="full_study"):
    # Split dataset into train and test based on test_size
    total_samples = len(df)
    split_idx = int((1 - test_size) * total_samples)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Extract features and target variables
    if 'target' in df.columns:
        target_col = 'target'
    else:
        target_col = df.columns[-1]
        logger.warning(f"Target column not found, using last column: {target_col}")

    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    feature_cols = [col for col in df.columns if col != target_col]
    x_train = train_df[feature_cols].values
    x_test = test_df[feature_cols].values
    cw_train = np.ones(len(y_train))

    train_stats = {
        "false_count": (y_train == 0).sum(),
        "true_count": (y_train == 1).sum(),
        "imbalance_ratio": (y_train == 0).sum() / max(1, (y_train == 1).sum())
    }
    test_stats = {
        "false_count": (y_test == 0).sum(),
        "true_count": (y_test == 1).sum(),
        "imbalance_ratio": (y_test == 0).sum() / max(1, (y_test == 1).sum())
    }
    data_stats = {
        "train": train_stats,
        "test": test_stats,
        "input_size": x_train.shape[1],
        "train_shape": x_train.shape,
        "test_shape": x_test.shape
    }
    if data_stats_extra:
        data_stats["data_params"] = data_stats_extra

    logger.info(f"Processing data with input size: {data_stats['input_size']}")
    logger.info(f"Train data shape: {data_stats['train_shape']}, Test data shape: {data_stats['test_shape']}")
    logger.info(f"Train labels: False {train_stats['false_count']} / True {train_stats['true_count']} (IR={train_stats['imbalance_ratio']:.3f})")
    logger.info(f"Test labels: False {test_stats['false_count']} / True {test_stats['true_count']} (IR={test_stats['imbalance_ratio']:.3f})")

    # --- Model grid search loop (Stage 1: Find best runner) ---
    lsensemble_config = model_config.get("lsensemble", {})
    stage1 = lsensemble_config.get("stage1", {})
    stage2 = lsensemble_config.get("stage2", {})

    # Stage 1 params (architecture)
    LS_num_experts = stage1.get("num_experts", [])
    LS_hidden_size = stage1.get("hidden_size", [])
    LS_drop_out = stage1.get("drop_out", [])
    LS_n_batch = stage1.get("n_batch", [])
    LS_n_epoch = stage1.get("n_epoch", [])
    lbfgs = stage1.get("lbfgs", False)
    mode = stage1.get("mode", "random")
    activation_fn = stage1.get("activation_fn", "relu")
    loss_fn = stage1.get("loss_fn", "F1")
    rb_each_expert = stage1.get("rb_each_expert", True)  # This is read from config

        # Stage 2 params - ALWAYS load these from the config file
    LS_alpha = stage2.get("alpha", [])
    LS_beta = stage2.get("beta", [])
    LS_Q_RB_C = stage2.get("Q_RB_C", [])
    LS_Q_RB_S = stage2.get("Q_RB_S", [])

    # Choose metric function
    if loss_fn.lower() == "custom":
        metric_fn = custom_metric
    elif loss_fn.lower() == "f1":
        metric_fn = lambda y_true, y_pred: f1_score(y_true, y_pred)
    elif loss_fn.lower() == "balanced_accuracy":
        metric_fn = lambda y_true, y_pred: balanced_accuracy_score(y_true, y_pred)
    else:
        metric_fn = lambda y_true, y_pred: balanced_accuracy_score(y_true, y_pred)  # default

    logger.info(f"Usando la métrica '{loss_fn}' para la evaluación de modelos.")

    n_simus = model_config.get("n_simus", 1)
    model_selection = model_config.get("model_selection", "conf")

    max_seconds_per_model = output_config.get("max_seconds_per_model", 300)

    # Before the model grid search loops:
    signal.signal(signal.SIGALRM, timeout_handler)

    # Stage 1: Find best runner (best config) for LSEnsemble, ignoring alpha/beta/qrbc/qrbs
    best_runner = None
    best_metric = -np.inf
    best_runner_config = None

    run_stage1 = True
    if study_mode == "stage2_only":
        run_stage1 = False
        logger.info("Study mode is 'stage2_only', skipping Stage 1.")
    
    if use_previous_best_runner and best_runner_file and os.path.exists(best_runner_file):
        run_stage1 = False
        with open(best_runner_file, "r") as f:
            best_runner = json.load(f)
        logger.info(f"Loaded best_runner from file, skipping Stage 1: {best_runner}")

    if run_stage1:
        # Build grid for all LSEnsemble params except alpha, beta, Q_RB_C, Q_RB_S
        runner_param_grid = list(product(
            LS_num_experts, LS_hidden_size, LS_drop_out, LS_n_batch, LS_n_epoch
        ))

        total_runner = len(runner_param_grid)
        logger.info(f"Starting Stage 1: Finding best runner from {total_runner} configurations.")
        for idx, runner_params in enumerate(runner_param_grid):
            num_experts, hidden_size, drop_out, n_batch, n_epoch = runner_params
            base_config = {
                'lbfgs': lbfgs,
                'mode': mode,
                'activation_fn': activation_fn,
                'loss_fn': loss_fn,
                'num_experts': num_experts,
                'hidden_size': hidden_size,
                'drop_out': drop_out,
                'n_batch': n_batch,
                'n_epoch': n_epoch,
                'input_size': data_stats['input_size'],
                'rb_each_expert': rb_each_expert  # Include the parameter in the model config
            }
            # Use default alpha/beta/qrbc/qrbs for runner selection (e.g. first value)
            #ALL set to "0" values for Fase 1
            alpha = 0.0
            beta = 0.0
            Q_RB_C = 2
            Q_RB_S = 1
            config = base_config.copy()
            config.update({'alpha': alpha, 'beta': beta, 'Q_RB_C': Q_RB_C, 'Q_RB_S': Q_RB_S})
            metric_vals = []
            for k_simu in range(n_simus):
                logger.info(f"Runner {idx+1}/{total_runner} Simu {k_simu+1}/{n_simus}: LSEnsemble with parameters: {config}")
                try:
                    signal.alarm(max_seconds_per_model)
                    start_time = time.time()
                    model = LSEnsemble(**config)
                    model.fit(x_train, y_train, sample_weight=cw_train)
                    ye_test = model.predict(x_test)
                    end_time = time.time()
                    signal.alarm(0)
                    time_taken = end_time - start_time
                    CM = confusion_matrix(y_test, ye_test)
                    if loss_fn.lower() == "custom":
                        metric, bac, minority_score = custom_metric(y_test, ye_test)
                    else:
                        metric = metric_fn(y_test, ye_test)
                        _, bac, minority_score = custom_metric(y_test, ye_test)
                    accuracy = np.mean(ye_test == y_test)
                    metric_vals.append(metric)
                    if writer is not None:
                        # Calcular porcentajes de etiquetas cambiadas
                        pos_switched = getattr(model, 'pos_to_neg_count', 0)
                        neg_switched = getattr(model, 'neg_to_pos_count', 0)
                        total_pos = getattr(model, 'total_pos', 1)  # Evitar división por cero
                        total_neg = getattr(model, 'total_neg', 1)  # Evitar división por cero

                        pct_pos_switched = (pos_switched / total_pos * 100) if total_pos > 0 else 0
                        pct_neg_switched = (neg_switched / total_neg * 100) if total_neg > 0 else 0

                        writer.writerow({
                            'data_params': str(data_stats.get('data_params', {})),
                            'model_name': 'LSEnsemble',
                            'params': str(config),
                            'metric': metric,
                            'confusion_matrix': str(CM.tolist()),
                            'accuracy': accuracy,
                            'time_taken': time_taken,
                            'label_switching_used': (alpha > 0) or (beta > 0),
                            'num_labels_switched_pos_to_neg': pos_switched,
                            'total_pos_labels': total_pos,
                            'pct_pos_switched': f"{pct_pos_switched:.2f}%",
                            'num_labels_switched_neg_to_pos': neg_switched,
                            'total_neg_labels': total_neg,
                            'pct_neg_switched': f"{pct_neg_switched:.2f}%",
                            'custom_metric': metric if loss_fn.lower() == "custom" else "",
                            'balanced_accuracy': bac,
                            'minority_score': minority_score
                        })
                        if csvfile is not None:
                            csvfile.flush()
                            os.fsync(csvfile.fileno())
                        logger.info(f"Wrote runner config result to CSV: {config}")
                except TimeoutException:
                    signal.alarm(0)
                    logger.warning(f"Timeout: Model run exceeded {max_seconds_per_model} seconds for config: {config}. Skipping.")
                    if writer is not None:
                        writer.writerow({
                            'data_params': str(data_stats.get('data_params', {})),
                            'model_name': 'LSEnsemble',
                            'params': str(config),
                            'metric': 'timeout',
                            'confusion_matrix': '',
                            'accuracy': '',
                            'time_taken': f'>{max_seconds_per_model}',
                            'label_switching_used': (alpha > 0) or (beta > 0),
                            'num_labels_switched_pos_to_neg': 0,
                            'total_pos_labels': 0,
                            'pct_pos_switched': "0.00%",
                            'num_labels_switched_neg_to_pos': 0,
                            'total_neg_labels': 0,
                            'pct_neg_switched': "0.00%",
                            'custom_metric': '',
                            'balanced_accuracy': '',
                            'minority_score': ''
                        })
                        if csvfile is not None:
                            csvfile.flush()
                            os.fsync(csvfile.fileno())
                    continue
            avg_metric = np.mean(metric_vals) if metric_vals else -np.inf
            if avg_metric > best_metric:
                best_metric = avg_metric
                best_runner = base_config.copy()
                best_runner_config = config.copy()

        logger.info(f"Best runner config from Stage 1: {best_runner}")
        # Guardar el best_runner encontrado
        if best_runner_file and best_runner:
            with open(best_runner_file, "w") as f:
                json.dump(best_runner, f)
            logger.info(f"Best runner saved to {best_runner_file}")

    if not best_runner:
        # Fallback to a default configuration if no best_runner is available after stage 1
        best_runner = {
            'lbfgs': lbfgs, 'mode': mode, 'activation_fn': activation_fn, 'loss_fn': loss_fn,
            'num_experts': LS_num_experts[0] if LS_num_experts else 21,
            'hidden_size': LS_hidden_size[0] if LS_hidden_size else 30,
            'drop_out': LS_drop_out[0] if LS_drop_out else 0.0,
            'n_batch': LS_n_batch[0] if LS_n_batch else 128,
            'n_epoch': LS_n_epoch[0] if LS_n_epoch else 50,
            'input_size': data_stats['input_size'],
            'rb_each_expert': rb_each_expert
        }
        logger.warning(f"No best_runner was found or loaded. Using default parameters for Stage 2: {best_runner}")
    
    # Stage 2: For the best runner, try all combinations of alpha, beta, Q_RB_C, Q_RB_S
    results = []
    total_lse = len(LS_alpha) * len(LS_beta) * len(LS_Q_RB_C) * len(LS_Q_RB_S)
    lse_idx = 0
    logger.info(f"Starting Stage 2: Testing {total_lse} label switching configurations.")
    for alpha in LS_alpha:
        for beta in LS_beta:
            for Q_RB_C in LS_Q_RB_C:
                for Q_RB_S in LS_Q_RB_S:
                    lse_idx += 1
                    config = best_runner.copy()
                    config.update({
                        'alpha': alpha,
                        'beta': beta,
                        'Q_RB_C': Q_RB_C,
                        'Q_RB_S': Q_RB_S
                    })
                    metric_vals = []
                    for k_simu in range(n_simus):
                        logger.info(f"LS {lse_idx}/{total_lse} Simu {k_simu+1}/{n_simus}: LSEnsemble with parameters: {config}")
                        try:
                            signal.alarm(max_seconds_per_model)
                            start_time = time.time()
                            model = LSEnsemble(**config)
                            model.fit(x_train, y_train, sample_weight=cw_train)
                            ye_test = model.predict(x_test)
                            end_time = time.time()
                            signal.alarm(0)
                            time_taken = end_time - start_time
                            CM = confusion_matrix(y_test, ye_test)
                            if loss_fn.lower() == "custom":
                                metric, bac, minority_score = custom_metric(y_test, ye_test)
                            else:
                                metric = metric_fn(y_test, ye_test)
                                _, bac, minority_score = custom_metric(y_test, ye_test)
                            accuracy = np.mean(ye_test == y_test)
                            metric_vals.append(metric)
                            label_switching_used = (alpha > 0) or (beta > 0)
                            num_labels_switched_pos_to_neg = getattr(model, 'pos_to_neg_count', 0)
                            num_labels_switched_neg_to_pos = getattr(model, 'neg_to_pos_count', 0)
                            total_pos = getattr(model, 'total_pos', 1)  # Evitar división por cero
                            total_neg = getattr(model, 'total_neg', 1)  # Evitar división por cero

                            pct_pos_switched = (num_labels_switched_pos_to_neg / total_pos * 100) if total_pos > 0 else 0
                            pct_neg_switched = (num_labels_switched_neg_to_pos / total_neg * 100) if total_neg > 0 else 0

                            if writer is not None:
                                writer.writerow({
                                    'data_params': str(data_stats.get('data_params', {})),
                                    'model_name': 'LSEnsemble',
                                    'params': str(config),
                                    'metric': metric,
                                    'confusion_matrix': str(CM.tolist()),
                                    'accuracy': accuracy,
                                    'time_taken': time_taken,
                                    'label_switching_used': label_switching_used,
                                    'num_labels_switched_pos_to_neg': num_labels_switched_pos_to_neg,
                                    'total_pos_labels': total_pos,
                                    'pct_pos_switched': f"{pct_pos_switched:.2f}%",
                                    'num_labels_switched_neg_to_pos': num_labels_switched_neg_to_pos,
                                    'total_neg_labels': total_neg,
                                    'pct_neg_switched': f"{pct_neg_switched:.2f}%",
                                    'custom_metric': metric if loss_fn.lower() == "custom" else "",
                                    'balanced_accuracy': bac,
                                    'minority_score': minority_score
                                })
                                if csvfile is not None:
                                    csvfile.flush()
                                    os.fsync(csvfile.fileno())
                                logger.info(f"Wrote LS config result to CSV: {config}")
                            logger.info(f"LS {lse_idx}/{total_lse} Simu {k_simu+1}/{n_simus}: Metric: {metric:.5f}, Accuracy: {accuracy:.5f}, Time: {time_taken:.2f}s")
                        except TimeoutException:
                            signal.alarm(0)
                            logger.warning(f"Timeout: Model run exceeded {max_seconds_per_model} seconds for config: {config}. Skipping.")
                            if writer is not None:
                                writer.writerow({
                                    'data_params': str(data_stats.get('data_params', {})),
                                    'model_name': 'LSEnsemble',
                                    'params': str(config),
                                    'metric': 'timeout',
                                    'confusion_matrix': '',
                                    'accuracy': '',
                                    'time_taken': f'>{max_seconds_per_model}',
                                    'label_switching_used': label_switching_used,
                                    'num_labels_switched_pos_to_neg': 0,
                                    'total_pos_labels': 0,
                                    'pct_pos_switched': "0.00%",
                                    'num_labels_switched_neg_to_pos': 0,
                                    'total_neg_labels': 0,
                                    'pct_neg_switched': "0.00%",
                                    'custom_metric': '',
                                    'balanced_accuracy': '',
                                    'minority_score': ''
                                })
                                if csvfile is not None:
                                    csvfile.flush()
                                    os.fsync(csvfile.fileno())
                            continue
                    avg_metric = np.mean(metric_vals) if metric_vals else -np.inf
                    results.append((config, avg_metric))

    # Al final busca el mejor resultado
    if results:
        best_config, best_metric = max(results, key=lambda x: x[1])
        logger.info(f"Best LSEnsemble config (with alpha/beta/qrbc/qrbs): {best_config}")
        logger.info(f"Best metric: {best_metric:.5f}")

    # Return best config for reference
    return best_config if results else best_runner_config

    # Stage 1: Find best runner (best config) for LSEnsemble, ignoring alpha/beta/qrbc/qrbs
    best_runner = None
    best_metric = -np.inf
    best_runner_config = None

    run_stage1 = True
    if study_mode == "stage2_only":
        run_stage1 = False
        logger.info("Study mode is 'stage2_only', skipping Stage 1.")
    
    if use_previous_best_runner and best_runner_file and os.path.exists(best_runner_file):
        run_stage1 = False
        with open(best_runner_file, "r") as f:
            best_runner = json.load(f)
        logger.info(f"Loaded best_runner from file, skipping Stage 1: {best_runner}")

    if run_stage1:
        # Build grid for all LSEnsemble params except alpha, beta, Q_RB_C, Q_RB_S
        runner_param_grid = list(product(
            LS_num_experts, LS_hidden_size, LS_drop_out, LS_n_batch, LS_n_epoch
        ))

        total_runner = len(runner_param_grid)
        logger.info(f"Starting Stage 1: Finding best runner from {total_runner} configurations.")
        for idx, runner_params in enumerate(runner_param_grid):
            num_experts, hidden_size, drop_out, n_batch, n_epoch = runner_params
            base_config = {
                'lbfgs': lbfgs,
                'mode': mode,
                'activation_fn': activation_fn,
                'loss_fn': loss_fn,
                'num_experts': num_experts,
                'hidden_size': hidden_size,
                'drop_out': drop_out,
                'n_batch': n_batch,
                'n_epoch': n_epoch,
                'input_size': data_stats['input_size'],
                'rb_each_expert': rb_each_expert  # Include the parameter in the model config
            }
            # Use default alpha/beta/qrbc/qrbs for runner selection (e.g. first value)
            #ALL set to "0" values for Fase 1
            alpha = 0.0
            beta = 0.0
            Q_RB_C = 2
            Q_RB_S = 1
            config = base_config.copy()
            config.update({'alpha': alpha, 'beta': beta, 'Q_RB_C': Q_RB_C, 'Q_RB_S': Q_RB_S})
            metric_vals = []
            for k_simu in range(n_simus):
                logger.info(f"Runner {idx+1}/{total_runner} Simu {k_simu+1}/{n_simus}: LSEnsemble with parameters: {config}")
                try:
                    signal.alarm(max_seconds_per_model)
                    start_time = time.time()
                    model = LSEnsemble(**config)
                    model.fit(x_train, y_train, sample_weight=cw_train)
                    ye_test = model.predict(x_test)
                    end_time = time.time()
                    signal.alarm(0)
                    time_taken = end_time - start_time
                    CM = confusion_matrix(y_test, ye_test)
                    if loss_fn.lower() == "custom":
                        metric, bac, minority_score = custom_metric(y_test, ye_test)
                    else:
                        metric = metric_fn(y_test, ye_test)
                        _, bac, minority_score = custom_metric(y_test, ye_test)
                    accuracy = np.mean(ye_test == y_test)
                    metric_vals.append(metric)
                    if writer is not None:
                        # Calcular porcentajes de etiquetas cambiadas
                        pos_switched = getattr(model, 'pos_to_neg_count', 0)
                        neg_switched = getattr(model, 'neg_to_pos_count', 0)
                        total_pos = getattr(model, 'total_pos', 1)  # Evitar división por cero
                        total_neg = getattr(model, 'total_neg', 1)  # Evitar división por cero

                        pct_pos_switched = (pos_switched / total_pos * 100) if total_pos > 0 else 0
                        pct_neg_switched = (neg_switched / total_neg * 100) if total_neg > 0 else 0

                        writer.writerow({
                            'data_params': str(data_stats.get('data_params', {})),
                            'model_name': 'LSEnsemble',
                            'params': str(config),
                            'metric': metric,
                            'confusion_matrix': str(CM.tolist()),
                            'accuracy': accuracy,
                            'time_taken': time_taken,
                            'label_switching_used': (alpha > 0) or (beta > 0),
                            'num_labels_switched_pos_to_neg': pos_switched,
                            'total_pos_labels': total_pos,
                            'pct_pos_switched': f"{pct_pos_switched:.2f}%",
                            'num_labels_switched_neg_to_pos': neg_switched,
                            'total_neg_labels': total_neg,
                            'pct_neg_switched': f"{pct_neg_switched:.2f}%",
                            'custom_metric': metric if loss_fn.lower() == "custom" else "",
                            'balanced_accuracy': bac,
                            'minority_score': minority_score
                        })
                        if csvfile is not None:
                            csvfile.flush()
                            os.fsync(csvfile.fileno())
                        logger.info(f"Wrote runner config result to CSV: {config}")
                except TimeoutException:
                    signal.alarm(0)
                    logger.warning(f"Timeout: Model run exceeded {max_seconds_per_model} seconds for config: {config}. Skipping.")
                    if writer is not None:
                        writer.writerow({
                            'data_params': str(data_stats.get('data_params', {})),
                            'model_name': 'LSEnsemble',
                            'params': str(config),
                            'metric': 'timeout',
                            'confusion_matrix': '',
                            'accuracy': '',
                            'time_taken': f'>{max_seconds_per_model}',
                            'label_switching_used': (alpha > 0) or (beta > 0),
                            'num_labels_switched_pos_to_neg': 0,
                            'total_pos_labels': 0,
                            'pct_pos_switched': "0.00%",
                            'num_labels_switched_neg_to_pos': 0,
                            'total_neg_labels': 0,
                            'pct_neg_switched': "0.00%",
                            'custom_metric': '',
                            'balanced_accuracy': '',
                            'minority_score': ''
                        })
                        if csvfile is not None:
                            csvfile.flush()
                            os.fsync(csvfile.fileno())
                    continue
            avg_metric = np.mean(metric_vals) if metric_vals else -np.inf
            if avg_metric > best_metric:
                best_metric = avg_metric
                best_runner = base_config.copy()
                best_runner_config = config.copy()

        logger.info(f"Best runner config from Stage 1: {best_runner}")
        # Guardar el best_runner encontrado
        if best_runner_file and best_runner:
            with open(best_runner_file, "w") as f:
                json.dump(best_runner, f)
            logger.info(f"Best runner saved to {best_runner_file}")

    if not best_runner:
        # Fallback to a default configuration if no best_runner is available after stage 1
        best_runner = {
            'lbfgs': lbfgs, 'mode': mode, 'activation_fn': activation_fn, 'loss_fn': loss_fn,
            'num_experts': LS_num_experts[0] if LS_num_experts else 21,
            'hidden_size': LS_hidden_size[0] if LS_hidden_size else 30,
            'drop_out': LS_drop_out[0] if LS_drop_out else 0.0,
            'n_batch': LS_n_batch[0] if LS_n_batch else 128,
            'n_epoch': LS_n_epoch[0] if LS_n_epoch else 50,
            'input_size': data_stats['input_size'],
            'rb_each_expert': rb_each_expert
        }
        logger.warning(f"No best_runner was found or loaded. Using default parameters for Stage 2: {best_runner}")
    
    # Stage 2: For the best runner, try all combinations of alpha, beta, Q_RB_C, Q_RB_S
    results = []
    total_lse = len(LS_alpha) * len(LS_beta) * len(LS_Q_RB_C) * len(LS_Q_RB_S)
    lse_idx = 0
    logger.info(f"Starting Stage 2: Testing {total_lse} label switching configurations.")

if __name__ == "__main__":
    setup_logging()
    model_config = {
        "model_selection": "conf",
        "n_simus": 1
    }
    output_config = {
        "csv_file": "test_results.csv",
        "log_level": "INFO"
    }
    # Example usage:
    # run_test_from_csv("your_dataset.csv", 0.2, model_config, output_config)
