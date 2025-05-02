#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import yaml
import logging
import importlib.util
import signal
import json
import datetime

# Simple script to run LSEnsemble tests with synthetic data
# Setup basic logging
logging.basicConfig(
    format="%(asctime)-15s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

BEST_RUNNER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_runner.txt")

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

def ask_use_previous_best():
    if os.path.exists(BEST_RUNNER_FILE):
        resp = input("¿Quieres usar el último best_runner guardado de la fase 1? (s/n): ").strip().lower()
        return resp == "s"
    return False

def backup_results_file(results_file_path):
    if os.path.exists(results_file_path):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = results_file_path.replace(
            ".csv", f"_{timestamp}.csv"
        )
        os.rename(results_file_path, backup_path)
        logger.info(f"Previous results file backed up as: {backup_path}")

def create_timestamped_results_dir(base_results_path):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.dirname(base_results_path)
    results_dir = os.path.join(base_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

class TeeLogger:
    def __init__(self, log_file_path):
        self.terminal = sys.__stdout__  # Use the original stdout, not sys.stdout
        self.log = open(log_file_path, "w", buffering=1, encoding="utf-8")
    def write(self, message):
        # Write to terminal and file, always flush after write
        if message:
            try:
                self.terminal.write(message)
                self.terminal.flush()
            except Exception:
                pass
            try:
                self.log.write(message)
                self.log.flush()
            except Exception:
                pass
    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass
        try:
            self.log.flush()
        except Exception:
            pass
    def close(self):
        try:
            self.log.flush()
            self.log.close()
        except Exception:
            pass

def main():
    # Set up signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize results to ensure it's always defined
    results = None

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
        "lsensemble": {
            "stage1": config["model"]["lsensemble"]["stage1"],
            "stage2": config["model"]["lsensemble"]["stage2"]
        }
    }
    
    # Create output configuration and ensure directory exists
    base_results_file_path = resolve_path(script_dir, config["output"]["results_file"])
    results_dir = create_timestamped_results_dir(base_results_file_path)
    results_file_path = os.path.join(results_dir, "test_results.csv")
    ensure_dir_exists(results_file_path)

    # Redirect stdout to both terminal and log file
    log_txt_path = os.path.join(results_dir, "terminal_output.txt")
    tee_logger = TeeLogger(log_txt_path)
    sys.stdout = tee_logger
    sys.stderr = tee_logger

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
    
    # Preguntar si se quiere usar el último best_runner guardado
    use_previous_best = ask_use_previous_best()

    # Run test_LSEnsemble with the dataset path, test_size and dataset parameters
    try:
        results = test_module.run_test_from_csv(
            dataset_path, test_size, model_config, output_config, dataset_params,
            use_previous_best_runner=use_previous_best, best_runner_file=BEST_RUNNER_FILE
        )
        logger.info(f"Testing completed, results saved to {output_config['csv_file']}")
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user.")
        logger.info("Results for completed model runs have been saved.")
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        logger.info("Partial results may have been saved.")
    finally:
        sys.stdout = tee_logger.terminal
        sys.stderr = tee_logger.terminal
        tee_logger.close()
    
    return results

if __name__ == "__main__":
    main()