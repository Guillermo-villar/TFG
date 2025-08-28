#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import yaml
import logging
from logging.handlers import RotatingFileHandler
import importlib.util
import signal
import json
import datetime

# Simple script to run LSEnsemble tests with synthetic data
# Setup basic logging - crearemos los handlers después de determinar el directorio de resultados
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def ask_user_level():
    """Ask user to select capilaridad level (1-3)"""
    print("\nSelect analysis level:")
    print("1 - Basic (fast, few combinations)")
    print("2 - Intermediate (balanced)")
    print("3 - Advanced (exhaustive, many combinations)")
    
    while True:
        try:
            level = int(input("Enter level (1-3): "))
            if level in [1, 2, 3]:
                return level
            else:
                print("Please enter 1, 2, or 3")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nOperation cancelled")
            sys.exit(0)

def build_config_from_level(base_config, level):
    """Build model configuration based on selected level"""
    level_config = base_config["capilaridad_levels"][level]
    
    model_config = {
        "model_selection": base_config["model"]["selection_method"],
        "n_simus": level_config["n_simulations"],
        "capilaridad_level": level,
        "max_seconds_per_model": level_config["max_seconds_per_model"],
        "n_simulations": level_config["n_simulations"],
        "mlpbayes": base_config["model"]["mlpbayes"],
        "lsensemble": {
            "stage1": level_config["stage1"],
            "stage2": level_config["stage2"]
        }
    }
    
    max_timeout = level_config["max_seconds_per_model"]
    
    logger.info(f"Using capilaridad level {level}")
    logger.info(f"Max timeout per model: {max_timeout}s")
    logger.info(f"Number of simulations: {model_config['n_simus']}")
    
    return model_config, max_timeout

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

def get_dataset_specific_best_runner_path(dataset_dir, dataset_id):
    """Get the path for a dataset-specific best_runner file in the dataset directory"""
    return os.path.join(dataset_dir, f"best_runner_{dataset_id}.json")

def create_run_directory_for_dataset(dataset_id, base_dir=None):
    """
    Create a timestamped run directory for a dataset and return paths.
    If base_dir is provided, it creates the run inside 'base_dir/runs/'.
    Otherwise, it uses the standard data_generator logic.
    """
    import datetime
    
    # Import data_generator module
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    import data_generator
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if base_dir:
        # This is the new logic for multiclass runs, creating a run inside the specified sub-directory
        run_dir = os.path.join(base_dir, "runs", timestamp)
        os.makedirs(run_dir, exist_ok=True)
        run_paths = {
            "run_dir": run_dir,
            "results_csv": os.path.join(run_dir, "test_results.csv"),
            "terminal_output": os.path.join(run_dir, "terminal_output.txt"),
            "timestamp": timestamp
        }
    else:
        # Fallback to original logic for standalone runs
        run_paths = data_generator.get_run_directory_structure(dataset_id, timestamp)
    
    # Ensure run directory exists
    ensure_dir_exists(run_paths["terminal_output"])  # This creates the run directory
    
    return run_paths

def ask_use_previous_best(best_runner_file):
    """
    Determine if we should use previous best_runner configuration.
    When running from GUI, this will automatically continue where it left off.
    When running from command line, it will still ask the user.
    """
    if os.path.exists(best_runner_file):
        # Check if running in a GUI context (no stdin available)
        try:
            # Try to check if stdin is available
            import sys
            if not sys.stdin.isatty():
                # Running in GUI context - automatically use previous configuration
                print("GUI mode detected: automatically continuing from previous study")
                return True
        except:
            # If any error checking stdin, assume GUI mode
            print("Non-interactive mode detected: automatically continuing from previous study")
            return True
        
        # Interactive mode - ask user as before
        try:
            with open(best_runner_file, 'r') as f:
                data = json.load(f)
            
            if "study_progress" in data:
                progress = data["study_progress"]
                stage1_complete = progress.get("stage1_completed", False)
                stage1_progress = f"{progress.get('completed_stage1_configs', 0)}/{progress.get('total_stage1_configs', 0)}"
                stage2_progress = f"{progress.get('completed_stage2_configs', 0)}/{progress.get('total_stage2_configs', 0)}"
                
                print(f"\nPrevious study found for this dataset:")
                print(f"  Stage 1 completed: {stage1_complete}")
                print(f"  Stage 1 progress: {stage1_progress}")
                print(f"  Stage 2 progress: {stage2_progress}")
                print(f"  Last updated: {progress.get('last_updated', 'Unknown')}")
                
                if not stage1_complete or progress.get('completed_stage2_configs', 0) < progress.get('total_stage2_configs', 0):
                    resp = input("¿Quieres continuar desde donde se quedó el estudio anterior? (s/n): ").strip().lower()
                else:
                    print("Previous study was completed.")
                    resp = input("¿Quieres usar la configuración óptima encontrada? (s/n): ").strip().lower()
            else:
                # Old format
                print("Previous best_runner found (old format).")
                resp = input("¿Quieres usar el último best_runner guardado de la fase 1? (s/n): ").strip().lower()
                
        except (json.JSONDecodeError, KeyError):
            print("Error reading previous study file.")
            resp = input("¿Quieres usar el último best_runner guardado? (s/n): ").strip().lower()
            
        return resp == "s"
    return False

def setup_logging(log_file_path, log_level="INFO"):
    """Configure logging to both console and file"""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))
    
    # Crear formato consistente para ambos handlers
    formatter = logging.Formatter("%(asctime)-15s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    # Eliminar handlers existentes
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para archivo
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Silenciar logging de matplotlib
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
    
    logger.info(f"Logs serán guardados en: {log_file_path}")

def main(level=None, study_mode="full_study", output_dir=None):
    # Set up signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize results to ensure it's always defined
    results = None

    # Get script directory and config file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    
    # Configuración inicial de logging básico (solo consola)
    logging.basicConfig(
        format="%(asctime)-15s %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Load configuration
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Failed to load config from {config_path}: {e}")
        sys.exit(1)
    
    # Ask user for level and build config (or use provided level)
    if level is None:
        level = ask_user_level()
    model_config, max_timeout = build_config_from_level(config, level)

    if study_mode == "stage2_only":
        logger.info("Study mode is 'stage2_only'. Skipping stage 1 iterations.")
        # Stage 2 only mode - the system will handle whether to use previous results
        # through the dataset status popup system

    
    # Import required modules
    data_generator, test_module = load_modules()
    
    # --- Dataset and Directory Setup ---
    dataset_path = None
    dataset_params = {}
    dataset_id = None
    dataset_dir = None

    if output_dir:
        # --- Multiclass Dichotomy Mode ---
        # The orchestrator (run_mc.py) has provided a specific directory for this run.
        logger.info(f"Running in multiclass dichotomy mode. Output will be saved to: {output_dir}")
        dataset_dir = output_dir
        
        # The dataset path is taken directly from the config, which run_mc updated.
        dataset_path = resolve_path(script_dir, config["data"]["external_dataset"])
        
        # The dataset_id is now based on the dichotomy's unique index to avoid collisions.
        dataset_id = config["data"].get("dichotomy_id", f"dichotomy_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
        
        dataset_params = {
            "source": "multiclass_dichotomy",
            "path": dataset_path,
            "dichotomy_id": dataset_id
        }
        logger.info(f"Dataset ID for this dichotomy: {dataset_id}")
        logger.info(f"Dataset directory for this dichotomy: {dataset_dir}")

    else:
        # --- Standard Standalone Mode ---
        # Check if we need to generate new data or use existing dataset
        if config["data"]["generate_new"]:
            logger.info("Generating new synthetic dataset")
            data_info = data_generator.generate_synthetic_data(params=config["data"]["params"])
            dataset_path = data_info['file_path']
            dataset_id = data_info['dataset_id']
            dataset_dir = data_info['dataset_dir']
            dataset_params = config["data"]["params"].copy()
            
            logger.info(f"Dataset generated and saved to: {dataset_path}")
            logger.info(f"Dataset ID: {dataset_id}")
            logger.info(f"Dataset directory: {dataset_dir}")
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
            original_dataset_path = resolve_path(script_dir, config["data"]["external_dataset"])
            data_info = data_generator.setup_real_dataset_structure(original_dataset_path)
            dataset_path = data_info['file_path']
            dataset_id = data_info['dataset_id']
            dataset_dir = data_info['dataset_dir']
            dataset_params = {
                "source": "external",
                "path": dataset_path,
                "original_path": data_info['original_path']
            }
            logger.info(f"External dataset copied to: {dataset_path}")
            logger.info(f"Dataset ID: {dataset_id}")
            logger.info(f"Dataset directory: {dataset_dir}")
    
    # Validate that the dataset file exists
    if not dataset_path or not os.path.exists(dataset_path):
        logger.error(f"Dataset file {dataset_path} does not exist")
        sys.exit(1)
    
    # Create a new run directory for this experiment inside the determined dataset_dir
    run_paths = create_run_directory_for_dataset(dataset_id, base_dir=dataset_dir)
    
    # Set up run-specific file paths
    results_file_path = run_paths["results_csv"]
    log_txt_path = run_paths["terminal_output"]
    
    # Configure logging to save in run directory
    setup_logging(log_txt_path, config["output"]["log_level"])
    
    # Initial message to verify logging works
    logger.info(f"Iniciando ejecución de LSEnsemble tests para dataset {dataset_id}")
    logger.info(f"Run directory: {run_paths['run_dir']}")
    logger.info(f"Run timestamp: {run_paths['timestamp']}")
    
    # Create output configuration
    output_config = {
        "csv_file": results_file_path,
        "log_level": config["output"]["log_level"],
        "max_seconds_per_model": max_timeout,
    }
    
    # Add test_size parameter from config
    test_size = config["data"]["params"].get("test_size", 0.2)
    
    # Log that we're starting to run tests with the dataset path and test_size
    logger.info(f"Starting model training and evaluation on dataset: {dataset_path}")
    logger.info(f"Using test_size: {test_size} (will use {int((1-test_size)*100)}% for training, {int(test_size*100)}% for testing)")
    
    # Get dataset-specific best_runner file path (stored at dataset level, not run level)
    dataset_best_runner_file = get_dataset_specific_best_runner_path(dataset_dir, dataset_id)
    logger.info(f"Using dataset-specific best_runner file: {dataset_best_runner_file}")
    
    # Ask if we should use the previous best_runner configuration
    # This will show the appropriate popup based on dataset status
    use_previous = ask_use_previous_best(dataset_best_runner_file)
    
    # Run test_LSEnsemble with the dataset path, test_size and dataset parameters
    try:
        results = test_module.run_test_from_csv(
            dataset_path, test_size, model_config, output_config, dataset_params,
            use_previous_best_runner=use_previous, best_runner_file=dataset_best_runner_file,
            study_mode=study_mode
        )
        logger.info(f"Testing completed, results saved to {output_config['csv_file']}")
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user.")
        logger.info("Results for completed model runs have been saved.")
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        logger.info("Partial results may have been saved.")
    
    # Final log message that will also appear in the file
    logger.info(f"Ejecución finalizada. Logs guardados en: {log_txt_path}")
    
    return results

if __name__ == "__main__":
    main()