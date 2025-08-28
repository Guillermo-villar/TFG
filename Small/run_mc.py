#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multiclass Experiment Runner
Orchestrates the execution of experiments on a set of binary dichotomies
generated from a multiclass dataset.
"""

import os
import sys
import yaml
import json
import logging
import datetime

# Ensure the current directory is in the path to import run_test
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import run_test
from data_generator import get_dataset_directory_structure, generate_real_data_id

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(levelname)s [MulticlassRunner]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def update_config_for_dichotomy(config_path, dichotomy_csv_path, dichotomy_index):
    """
    Updates the main config.yaml to point to the correct dichotomy CSV for the next run.
    It sets generate_new to False, updates the external_dataset path, and adds a
    unique index to ensure the dichotomy run is treated as unique.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if 'data' not in config:
            config['data'] = {}

        config['data']['generate_new'] = False
        config['data']['external_dataset'] = dichotomy_csv_path
        # Add a unique identifier for this specific dichotomy run
        config['data']['dichotomy_id'] = f"dichotomy_{dichotomy_index:02d}"

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Updated config.yaml for dichotomy index {dichotomy_index}")
        return True
    except Exception as e:
        logger.error(f"Failed to update config.yaml for dichotomy {dichotomy_csv_path}: {e}")
        return False

def get_multiclass_progress_path(main_dataset_dir, dataset_id):
    """Generates the path for the multiclass progress file inside the main dataset directory."""
    return os.path.join(main_dataset_dir, f"multiclass_progress_{dataset_id}.json")

def get_multiclass_best_runner_path(main_dataset_dir, dataset_id):
    """Generates the path for the multiclass best runner file."""
    return os.path.join(main_dataset_dir, f"multiclass_best_runner_{dataset_id}.json")

def save_multiclass_progress(progress_file, progress_data):
    """Saves the overall progress of the multiclass experiment."""
    try:
        progress_data["last_updated"] = datetime.datetime.now().isoformat()
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        logger.info(f"Multiclass progress saved to {progress_file}")
    except Exception as e:
        logger.error(f"Error saving multiclass progress to {progress_file}: {e}")

def load_multiclass_progress(progress_file):
    """Loads the overall progress of the multiclass experiment."""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            logger.info(f"Loaded existing multiclass progress from {progress_file}")
            return progress_data
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not load or parse multiclass progress file {progress_file}. Starting fresh. Error: {e}")
            return None
    return None

def main(dichotomy_info, level, study_mode):
    """
    Main function to run the multiclass experiment.

    Parameters:
    -----------
    dichotomy_info : dict
        The dictionary returned by create_multiclass_dichotomies.
    level : int
        The capilaridad level (1-3).
    study_mode : str
        The study mode ('full_study' or 'stage2_only').
    """
    main_dataset_id = dichotomy_info.get('dataset_id')
    if not main_dataset_id:
        logger.error("Dichotomy information is missing 'dataset_id'. Aborting.")
        return

    # This is the main folder for the entire multiclass experiment
    main_dataset_paths = get_dataset_directory_structure(main_dataset_id)
    main_dataset_dir = main_dataset_paths["dataset_dir"]
    os.makedirs(main_dataset_dir, exist_ok=True)

    logger.info(f"--- Starting Multiclass Experiment for Dataset ID: {main_dataset_id} ---")
    logger.info(f"Main experiment directory: {main_dataset_dir}")
    logger.info(f"Strategy: {dichotomy_info.get('strategy', 'N/A').upper()}, Level: {level}, Mode: {study_mode}")

    config_path = os.path.join(script_dir, "config.yaml")
    progress_file = get_multiclass_progress_path(main_dataset_dir, main_dataset_id)
    best_runner_file = get_multiclass_best_runner_path(main_dataset_dir, main_dataset_id)

    # Load or initialize progress
    progress_data = load_multiclass_progress(progress_file)
    if not progress_data:
        progress_data = {
            "dataset_id": main_dataset_id,
            "strategy": dichotomy_info.get('strategy'),
            "n_classes": dichotomy_info.get('n_classes'),
            "class_names": dichotomy_info.get('class_names'),
            "code_matrix": dichotomy_info.get('code_matrix'),
            "dichotomies": {
                os.path.basename(dich['file']): {"status": "pending", "output_dir": None, "best_metric": None}
                for dich in dichotomy_info.get('dichotomies', [])
            },
            "last_updated": None
        }

    # Load or initialize the best runner file for the whole multiclass experiment
    if os.path.exists(best_runner_file):
        with open(best_runner_file, 'r') as f:
            multiclass_best_runner = json.load(f)
    else:
        multiclass_best_runner = {
            "best_metric_so_far": -1,
            "best_runner_config": None,
            "source_dichotomy": None,
            "last_updated": None
        }

    dichotomies_to_run = dichotomy_info.get('dichotomies', [])
    total_dichotomies = len(dichotomies_to_run)
    
    for i, dichotomy in enumerate(dichotomies_to_run):
        dichotomy_index = i + 1
        dichotomy_file_path = dichotomy['file']
        dichotomy_name = os.path.basename(dichotomy_file_path)
        
        logger.info("-" * 60)
        logger.info(f"Processing Dichotomy {dichotomy_index}/{total_dichotomies}: {dichotomy_name}")

        # Check progress
        if progress_data["dichotomies"].get(dichotomy_name, {}).get("status") == "completed":
            logger.info(f"Dichotomy {dichotomy_name} already marked as completed. Skipping.")
            continue

        # Create a dedicated output directory for this dichotomy run
        dichotomy_output_dir = os.path.join(main_dataset_dir, f"dichotomy_{dichotomy_index:02d}_run")
        os.makedirs(dichotomy_output_dir, exist_ok=True)
        
        # Update config to be unique for this dichotomy
        if not update_config_for_dichotomy(config_path, dichotomy_file_path, dichotomy_index):
            logger.error(f"Could not update config for {dichotomy_name}. Skipping this dichotomy.")
            progress_data["dichotomies"][dichotomy_name]["status"] = "error_config"
            continue

        try:
            # Mark as in-progress
            progress_data["dichotomies"][dichotomy_name]["status"] = "in_progress"
            progress_data["dichotomies"][dichotomy_name]["output_dir"] = dichotomy_output_dir
            save_multiclass_progress(progress_file, progress_data)

            # Run the single-dichotomy experiment, passing the dedicated output directory
            # This requires run_test.main to be modified to accept 'output_dir'
            best_runner_from_dichotomy = run_test.main(
                level=level, 
                study_mode=study_mode,
                output_dir=dichotomy_output_dir
            )

            # Mark as completed
            progress_data["dichotomies"][dichotomy_name]["status"] = "completed"
            logger.info(f"Successfully completed experiment for dichotomy {dichotomy_name}")

            # Update the overall best runner if this one is better
            if best_runner_from_dichotomy and best_runner_from_dichotomy.get("best_metric_so_far", -1) > multiclass_best_runner["best_metric_so_far"]:
                logger.info(f"New best multiclass runner found from dichotomy {dichotomy_name}!")
                logger.info(f"Metric improved from {multiclass_best_runner['best_metric_so_far']:.4f} to {best_runner_from_dichotomy['best_metric_so_far']:.4f}")
                
                multiclass_best_runner["best_metric_so_far"] = best_runner_from_dichotomy["best_metric_so_far"]
                multiclass_best_runner["best_runner_config"] = best_runner_from_dichotomy["best_runner"]
                multiclass_best_runner["source_dichotomy"] = dichotomy_name
                multiclass_best_runner["last_updated"] = datetime.datetime.now().isoformat()
                
                with open(best_runner_file, 'w') as f:
                    json.dump(multiclass_best_runner, f, indent=2)
                logger.info(f"Saved new best multiclass runner to {best_runner_file}")

            if best_runner_from_dichotomy:
                 progress_data["dichotomies"][dichotomy_name]["best_metric"] = best_runner_from_dichotomy.get("best_metric_so_far")


        except KeyboardInterrupt:
            logger.warning("Multiclass experiment interrupted by user. Saving progress.")
            progress_data["dichotomies"][dichotomy_name]["status"] = "interrupted"
            save_multiclass_progress(progress_file, progress_data)
            sys.exit(0) # Exit cleanly
        except Exception as e:
            logger.error(f"An error occurred while processing dichotomy {dichotomy_name}: {e}", exc_info=True)
            progress_data["dichotomies"][dichotomy_name]["status"] = "error_runtime"
        
        finally:
            # Save progress after each dichotomy is processed
            save_multiclass_progress(progress_file, progress_data)

    logger.info(f"--- Multiclass Experiment for Dataset ID: {main_dataset_id} Finished ---")
    logger.info(f"Overall best runner saved in {best_runner_file}")

if __name__ == "__main__":
    logger.info("This script is intended to be called from the GUI or another orchestrator, not run directly.")
    logger.info("It requires a 'dichotomy_info' dictionary to be passed to its main function.")
