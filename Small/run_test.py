#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import yaml
import logging
import importlib.util
import signal

# Simple script to run LSEnsemble tests with synthetic data
# Setup basic logging
logging.basicConfig(
    format="%(asctime)-15s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def load_modules():
    """Import required modules"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Import data_generator
    try:
        spec = importlib.util.spec_from_file_location(
            "data_generator", 
            os.path.join(script_dir, "data_generator.py")
        )
        data_generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_generator)
    except Exception as e:
        logger.error(f"Failed to import data_generator module: {e}")
        sys.exit(1)
        
    # Import test_LSEnsemble
    try:
        spec = importlib.util.spec_from_file_location(
            "test_LSEnsemble", 
            os.path.join(script_dir, "test_LSEnsemble.py")
        )
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
    except Exception as e:
        logger.error(f"Failed to import test_LSEnsemble module: {e}")
        sys.exit(1)
        
    return data_generator, test_module

def resolve_path(base_dir, path):
    """
    Convert a relative path to absolute path based on the script directory
    """
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)

def ensure_dir_exists(file_path):
    """
    Ensure the directory for the specified file path exists
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def signal_handler(sig, frame):
    """Handle Ctrl+C to ensure clean shutdown"""
    logger.info("\nProcess interrupted by user. Shutting down gracefully...")
    logger.info("Any completed model runs have been saved to the results file.")
    sys.exit(0)

def main():
    # Set up signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)

    # Get script directory and config file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    
    # Load configuration
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        sys.exit(1)
    
    # Import required modules
    data_generator, test_module = load_modules()
    
    # Create model configuration dictionary compatible with test_LSEnsemble
    model_config = {
        "model_selection": config["model"]["selection_method"],
        "n_simus": config["model"]["n_simulations"],
        "mlpbayes": config["model"]["mlpbayes"],
        "lsensemble": config["model"]["lsensemble"]
    }
    
    # Create output configuration and ensure directory exists
    results_file_path = resolve_path(script_dir, config["output"]["results_file"])
    ensure_dir_exists(results_file_path)
    
    output_config = {
        "csv_file": results_file_path,
        "log_level": config["output"]["log_level"],
        "max_seconds_per_model": config["model"].get("max_seconds_per_model", None),
    }
    
    # Set dataset path (single CSV file)
    dataset_path = None
    dataset_params = {}
    
    # Check if we need to generate new data or use existing dataset
    if config["data"]["generate_new"]:
        logger.info("Generating new synthetic dataset")
        
        # Resolve path in data config
        dataset_file = resolve_path(script_dir, config["data"]["dataset_file"])
        ensure_dir_exists(os.path.dirname(dataset_file))  # <-- fix here
        
        # Save the dataset parameters for logging
        dataset_params = config["data"]["params"].copy()
        
        # Generate the dataset
        data_info = data_generator.generate_synthetic_data(
            params=config["data"]["params"],
            output_file=dataset_file
        )
        
        # Get the generated dataset path
        dataset_path = data_info['file_path']
        
        # Log dataset statistics
        logger.info(f"Dataset generated and saved to: {dataset_path}")
        logger.info("Dataset statistics:")
        for metric, value in data_info["stats"].items():
            if isinstance(value, dict):
                logger.info(f"  {metric}:")
                for submetric, subvalue in value.items():
                    logger.info(f"    {submetric}: {subvalue}")
            else:
                logger.info(f"  {metric}: {value}")
    else:
        logger.info("Using external dataset from config")
        
        # Use dataset path from external_dataset
        dataset_path = resolve_path(script_dir, config["data"]["external_dataset"]["train_data"])
        
        # Add information about the external dataset
        dataset_params = {
            "source": "external",
            "path": dataset_path
        }
    
    # Validate that the dataset file exists
    if not dataset_path or not os.path.exists(dataset_path):
        logger.error(f"Dataset file {dataset_path} does not exist")
        sys.exit(1)
    
    # Add test_size parameter from config
    test_size = config["data"]["params"].get("test_size", 0.2)
    
    # Log that we're starting to run tests with the dataset path and test_size
    logger.info(f"Starting model training and evaluation on dataset: {dataset_path}")
    logger.info(f"Using test_size: {test_size} (will use {int((1-test_size)*100)}% for training, {int(test_size*100)}% for testing)")
    
    # Run test_LSEnsemble with the dataset path, test_size and dataset parameters
    try:
        results = test_module.run_test_from_csv(
            dataset_path, test_size, model_config, output_config, dataset_params
        )
        logger.info(f"Testing completed, results saved to {output_config['csv_file']}")
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user.")
        logger.info("Results for completed model runs have been saved.")
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        logger.info("Partial results may have been saved.")
    
    return results

if __name__ == "__main__":
    main()