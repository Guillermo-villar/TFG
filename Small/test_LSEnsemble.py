#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import csv
import time
import os
import numpy as np
import pandas as pd
# Combine all combinations of dynamic parameters
from itertools import product
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from uc3m.mlpbayesian import MLPBayesBinW
from uc3m.labelswitching import LSEnsemble

# -----------------------------------------------------------------------------
# INITIALIZE LOGGER
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

def setup_logging(log_level="INFO"):
    """Set up logging with the specified level"""
    logging.getLogger().setLevel(getattr(logging, log_level))
    FORMAT = "%(asctime)-15s %(message)s"
    logging.basicConfig(
        format=FORMAT, level=getattr(logging, log_level), datefmt="%Y-%m-%d %H:%M:%S"
    )
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)

def run_test_from_csv(dataset_path, test_size, model_config, output_config, dataset_params=None):
    """
    Run the ensemble test with data loaded from a single CSV file.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the CSV file containing the dataset
    test_size : float
        Proportion of the dataset to include in the test split (between 0 and 1)
    model_config : dict
        Configuration for model parameters
    output_config : dict
        Output configuration
    dataset_params : dict, optional
        Parameters used to generate the dataset (will be logged in the CSV)
        
    Returns:
    --------
    saved_model : dict or list
        The best model(s) according to the selection strategy
    """
    # Setup logging
    setup_logging(output_config.get("log_level", "INFO"))
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Split dataset into train and test based on test_size
    total_samples = len(df)
    split_idx = int((1 - test_size) * total_samples)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    logger.info(f"Split dataset with test_size={test_size}: {len(train_df)} training samples, {len(test_df)} test samples")
    
    # Extract features and target variables
    # First determine which columns are features vs target
    if 'target' in df.columns:
        target_col = 'target'
    else:
        # Assume last column is target if not explicitly named
        target_col = df.columns[-1]
        logger.warning(f"Target column not found, using last column: {target_col}")
    
    # Extract targets
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    
    # Extract features (all columns except target)
    feature_cols = [col for col in df.columns if col != target_col]
    x_train = train_df[feature_cols].values
    x_test = test_df[feature_cols].values
    
    # Use unit weights for training
    logger.info("Using unit weights for training")
    cw_train = np.ones(len(y_train))
    
    # Calculate statistics
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
    
    # Add dataset parameters to data_stats
    if dataset_params:
        data_stats["data_params"] = dataset_params
    
    # Log data statistics
    logger.info(f"Processing data with input size: {data_stats['input_size']}")
    logger.info(f"Train data shape: {data_stats['train_shape']}, Test data shape: {data_stats['test_shape']}")
    logger.info(f"Train labels: False {train_stats['false_count']} / True {train_stats['true_count']} (IR={train_stats['imbalance_ratio']:.3f})")
    logger.info(f"Test labels: False {test_stats['false_count']} / True {test_stats['true_count']} (IR={test_stats['imbalance_ratio']:.3f})")
    
    # Run the model training and evaluation
    return run_model_evaluation(
        x_train, y_train, x_test, y_test, cw_train, data_stats,
        model_config, output_config
    )

def run_test_from_csv_legacy(dataset_paths, model_config, output_config):
    """
    Run the ensemble test with data loaded from CSV files (legacy method).
    
    Parameters:
    -----------
    dataset_paths : dict
        Paths to the CSV files with the dataset
        - 'train_data': path to training data CSV
        - 'test_data': path to test data CSV
        - 'weights': path to weights CSV (optional)
        - 'metadata': path to metadata CSV (optional)
    model_config : dict
        Configuration for model parameters
    output_config : dict
        Output configuration
        
    Returns:
    --------
    saved_model : dict or list
        The best model(s) according to the selection strategy
    """
    # Setup logging
    setup_logging(output_config.get("log_level", "INFO"))
    
    # Check if separate train/test files are provided
    train_data_path = dataset_paths.get('train_data')
    test_data_path = dataset_paths.get('test_data')
    weights_path = dataset_paths.get('weights')
    
    if not train_data_path or not test_data_path:
        raise ValueError("Both train_data and test_data paths are required")
    
    # Load training data
    logger.info(f"Loading training data from {train_data_path}")
    train_df = pd.read_csv(train_data_path)
    
    # Load test data
    logger.info(f"Loading test data from {test_data_path}")
    test_df = pd.read_csv(test_data_path)
    
    # Extract features and target variables
    # First determine which columns are features vs target
    if 'target' in train_df.columns:
        target_col = 'target'
    else:
        # Assume last column is target if not explicitly named
        target_col = train_df.columns[-1]
        logger.warning(f"Target column not found, using last column: {target_col}")
    
    # Extract targets
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    
    # Extract features (all columns except target)
    feature_cols = [col for col in train_df.columns if col != target_col]
    x_train = train_df[feature_cols].values
    x_test = test_df[feature_cols].values
    
    # Load weights if provided
    if weights_path:
        logger.info(f"Loading weights from {weights_path}")
        try:
            weights_df = pd.read_csv(weights_path)
            # If weights file has a single column, use that
            if len(weights_df.columns) == 1:
                cw_train = weights_df.iloc[:, 0].values
            else:
                # Look for a column named 'weight' or similar
                weight_col = next((col for col in weights_df.columns if 'weight' in col.lower()), None)
                if weight_col:
                    cw_train = weights_df[weight_col].values
                else:
                    # Use first column as weights if no specific weight column found
                    logger.warning("No weight column identified, using first column as weights")
                    cw_train = weights_df.iloc[:, 0].values
                    
            # Ensure weights array has same length as training data
            if len(cw_train) != len(y_train):
                logger.warning(f"Weight array length ({len(cw_train)}) doesn't match training data length ({len(y_train)})")
                # Trim or extend weights as needed
                if len(cw_train) > len(y_train):
                    cw_train = cw_train[:len(y_train)]
                else:
                    # Extend with ones
                    cw_train = np.pad(cw_train, (0, len(y_train) - len(cw_train)), 'constant', constant_values=1)
        except Exception as e:
            logger.error(f"Error loading weights: {e}. Using unit weights.")
            cw_train = np.ones(len(y_train))
    else:
        # Use unit weights if no weights file provided
        logger.info("No weights file provided, using unit weights")
        cw_train = np.ones(len(y_train))
    
    # Calculate statistics
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
    
    # Log data statistics
    logger.info(f"Processing data with input size: {data_stats['input_size']}")
    logger.info(f"Train data shape: {data_stats['train_shape']}, Test data shape: {data_stats['test_shape']}")
    logger.info(f"Train labels: False {train_stats['false_count']} / True {train_stats['true_count']} (IR={train_stats['imbalance_ratio']:.3f})")
    logger.info(f"Test labels: False {test_stats['false_count']} / True {test_stats['true_count']} (IR={test_stats['imbalance_ratio']:.3f})")
    
    # Run the model training and evaluation
    return run_model_evaluation(
        x_train, y_train, x_test, y_test, cw_train, data_stats,
        model_config, output_config
    )

def run_model_evaluation(x_train, y_train, x_test, y_test, cw_train, data_stats, model_config, output_config):
    """
    Run model training and evaluation with the provided data.
    
    Parameters:
    -----------
    x_train : numpy.ndarray
        Training data features
    y_train : numpy.ndarray
        Training data labels
    x_test : numpy.ndarray
        Test data features
    y_test : numpy.ndarray
        Test data labels
    cw_train : numpy.ndarray
        Class weights for training data
    data_stats : dict
        Dictionary with statistics about the data
    model_config : dict
        Configuration for model parameters
    output_config : dict
        Output configuration
        
    Returns:
    --------
    saved_model : dict or list
        The best model(s) according to the selection strategy
    """
    # Initialize best_model_overall
    best_model_overall = [None, None, 0, None]
    
    # Initialize best_model_conf
    best_model_conf = dict()
    
    # ------------------------------------------------------------------------------
    # Simulation Params
    # ------------------------------------------------------------------------------
    
    # Model selection strategy
    model_selection = model_config.get("model_selection", "conf")
    
    # Number of simulations per configuration
    n_simus = model_config.get("n_simus", 1)
    
    # MLPBayes parameters
    mlpbayes_config = model_config.get("mlpbayes", {})
    
    # Fixed value arguments
    costeN = mlpbayes_config.get("costeN", 1)
    costeP = mlpbayes_config.get("costeP", 1)
    
    s_No = mlpbayes_config.get("s_No", [1])
    s_NnBase = mlpbayes_config.get("s_NnBase", [20, 30])
    s_pDOent = mlpbayes_config.get("s_pDOent", [0.0, 0.1])
    s_pDOocu = mlpbayes_config.get("s_pDOocu", [0.0, 0.1])
    s_tActBase = mlpbayes_config.get("s_tActBase", ["tanh", "relu"])
    s_tActSalida = mlpbayes_config.get("s_tActSalida", ["identity"])
    s_nBatch = mlpbayes_config.get("s_nBatch", [128, 256])
    s_nEpoch = mlpbayes_config.get("s_nEpoch", [100, 200])
    
    # LSEnsemble parameters
    lsensemble_config = model_config.get("lsensemble", {})
    
    LS_alpha = lsensemble_config.get("alpha", [0.0, 0.05, 0.1])
    LS_beta = lsensemble_config.get("beta", [0.0, 0.05, 0.1])
    LS_Q_RB_C = lsensemble_config.get("Q_RB_C", [2.0, 5.0])
    LS_Q_RB_S = lsensemble_config.get("Q_RB_S", [1, 2, 5])
    LS_num_experts = lsensemble_config.get("num_experts", [21, 31])
    LS_hidden_size = lsensemble_config.get("hidden_size", [30, 40])
    LS_drop_out = lsensemble_config.get("drop_out", [0, 0.1])
    LS_n_batch = lsensemble_config.get("n_batch", [128, 256])
    LS_n_epoch = lsensemble_config.get("n_epoch", [50, 100])
    
    # LSEnsemble static parameters
    lbfgs = lsensemble_config.get("lbfgs", False)
    mode = lsensemble_config.get("mode", "random")
    activation_fn = lsensemble_config.get("activation_fn", "relu")
    loss_fn = lsensemble_config.get("loss_fn", "F1")
    
    # Define model configurations
    model_configs = [
        {
            "name": "LSEnsemble",
            "class": LSEnsemble,
            "params": {
                "lbfgs": lbfgs,
                "mode": mode, 
                "activation_fn": activation_fn,
                "loss_fn": loss_fn,
            },
            'dynamic_params': list(product(LS_Q_RB_S, LS_Q_RB_C, LS_num_experts, LS_hidden_size, LS_drop_out, LS_n_batch, LS_n_epoch))
        },
        {
            "name": "MLPBayesBinW",
            "class": MLPBayesBinW,
            "params": {
                "class_cost": [costeN, costeP],
            },
            'dynamic_params': list(product(s_No, s_NnBase, s_tActBase, s_tActSalida, s_pDOent, s_pDOocu, s_nBatch, s_nEpoch))
        }
    ]
    
    # Results dictionary to track metrics
    results_by_model = {}
    for model_config in model_configs:
        model_name = model_config["name"]
        results_by_model[model_name] = []
        
    # Get CSV output path
    csv_file = output_config.get("csv_file", "test_results.csv")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_file) if os.path.dirname(csv_file) else '.', exist_ok=True)
    
    # CSV column headers
    csv_columns = ['data_params', 'model_name', 'params', 'metric', 'confusion_matrix', 'accuracy', 'time_taken']
    
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        
        # Iterate through different versions of alpha and beta parameters
        for alpha in LS_alpha:
            for beta in LS_beta:
                for model_config in model_configs:
                    if model_config['name'] == 'LSEnsemble':
                        model_config['dynamic_params'] = list(product([alpha], [beta], LS_Q_RB_S, LS_Q_RB_C, LS_num_experts, LS_hidden_size, LS_drop_out, LS_n_batch, LS_n_epoch))
                    
                    model_name = model_config['name']
                    model_class = model_config['class']
                    param_grid = model_config['params']
                    CV_config = []
                    dynamic_combinations = model_config['dynamic_params']
                    n_conf_test = len(dynamic_combinations)
                    metric_conf = np.zeros((n_conf_test, n_simus))
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
                                'input_size': data_stats['input_size']
                            })
                        CV_config.append(updated_config)
                    
                    k_conf = 0
                    for cv_config in CV_config:
                        for k_simu in range(n_simus):
                            start_time = time.time()
                            model = model_class(**cv_config)
                            logger.info(f'Running model: {model_name} with parameters: {cv_config}')
                            model.fit(x_train, y_train, sample_weight=cw_train)
                            ye_test = model.predict(x_test)
                            end_time = time.time()
                            time_taken = end_time - start_time
                            CM = confusion_matrix(y_test, ye_test)
                            metric = balanced_accuracy_score(y_test, ye_test)
                            accuracy = np.mean(ye_test == y_test)
                            metric_conf[k_conf, k_simu] = metric
                            
                            # Log current model performance
                            logger.info(f'Current model: {model_name} with parameters: {cv_config}, Metric: {metric:.5f}, Accuracy: {accuracy:.5f}, Time taken: {time_taken:.2f} seconds')
                            
                            # Update best_model_overall during model evaluation
                            if metric > best_model_overall[2]:
                                best_model_overall = [model, cv_config, metric, CM]
                                logger.info(f'New best configuration overall: {cv_config}')
                                logger.info(f'New best metric overall: {best_model_overall[2]:.5f}')
                            
                            # Update best_model_conf
                            if model_name not in best_model_conf:
                                best_model_conf[model_name] = []
                            
                            results_by_model[model_name].append((model, cv_config, metric, CM))
                            
                            writer.writerow({
                                'data_params': str(data_stats.get('data_params', {})),
                                'model_name': model_name,
                                'params': str(cv_config),
                                'metric': metric,
                                'confusion_matrix': str(CM.tolist()),
                                'accuracy': accuracy,
                                'time_taken': time_taken
                            })
                            
                        # Calculate average performance for this configuration
                        avg_metric = np.mean(metric_conf[k_conf])
                        best_model_conf.setdefault(model_name, []).append([None, cv_config, avg_metric, None])
                        k_conf += 1
    
    # Report best model and performance
    logger.info(f"Best overall model: {best_model_overall[0].__class__.__name__}")
    logger.info(f"Best overall configuration: {best_model_overall[1]}")
    logger.info(f"Best overall metric: {best_model_overall[2]:.5f}")
    
    for model_config in model_configs:
        model_name = model_config["name"]
        logger.info(f"Model: {model_name}")
        logger.info("-----------------------------------------------------")
        logger.info('Peak Performance')
        logger.info(f"Selected configuration {best_model_overall[1]}")
        logger.info('Best Peak performance: BAcc = {:.5f}'.format(best_model_overall[2]))
        logger.info("-----------------------------------------------------")
        
        if model_name in best_model_conf:
            # Get the configuration with the best average performance
            if best_model_conf[model_name]:  # Check if not empty
                idx_best = np.argmax([conf[2] for conf in best_model_conf[model_name]])
                model_conf = best_model_conf[model_name][idx_best]
                logger.info('Average Performance')
                logger.info(f"Selected configuration {model_conf[1]}")
                logger.info('Best Average performance: BAcc = {:.5f}'.format(model_conf[2]))
    
    # Return best model according to selection strategy
    if model_selection == "ind":
        saved_model = best_model_overall
    elif model_selection == "conf":
        saved_models = {}
        for model_name in best_model_conf:
            if best_model_conf[model_name]:  # Check if not empty
                idx_best = np.argmax([conf[2] for conf in best_model_conf[model_name]])
                saved_models[model_name] = best_model_conf[model_name][idx_best]
        saved_model = saved_models
    else:
        raise TypeError('Unexpected value for "model_selection"')
    
    return saved_model


if __name__ == "__main__":
    # If run directly, use default parameters
    setup_logging()
    
    # Default model configuration
    model_config = {
        "model_selection": "conf",
        "n_simus": 1
    }
    
    # Default output configuration
    output_config = {
        "csv_file": "test_results.csv",
        "log_level": "INFO"
    }
