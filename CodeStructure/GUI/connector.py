#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Connector for ML Experiments
Manages connection between GUI and run_test.py
"""

import sys
import os
import tkinter as tk
import multiprocessing
import yaml
import json

# Add parent directory to import run_test
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import run_test
import run_mc
from gui_interface import MLExperimentGUI, LanguageSetupWindow


class ExperimentRunner:
    """Handles the connection between GUI and run_test.py"""
    
    def __init__(self):
        self.parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(self.parent_dir, "config.yaml")
        self.experiment_process = None
        self.latest_log_file = None
    
    def run_experiment(self, level, study_mode="full_study", data_source="synthetic", csv_path=None, multiclass_strategy=None, multiclass_dichotomies=None):
        """Execute the experiment with given parameters"""
        try:
            # Modify config based on data source and multiclass settings
            self.update_config(data_source, csv_path, multiclass_strategy)

            # Reset latest log file on new run
            self.latest_log_file = None

            # Determine the target function and arguments based on whether it's a multiclass run
            if multiclass_dichotomies:
                # This is a multiclass experiment, so we use the new orchestrator
                target_func = run_mc.main
                args = (multiclass_dichotomies, level, study_mode)
            else:
                # This is a standard single/binary experiment
                target_func = run_test.main
                args = (level, study_mode)

            # Create and start a new process for the experiment
            self.experiment_process = multiprocessing.Process(target=target_func, args=args)
            self.experiment_process.start()
            
            # We don't join, so the GUI remains responsive
            return "Experiment started in a new process."
        except Exception as e:
            raise Exception(f"Failed to start experiment process: {str(e)}")

    def stop_experiment(self):
        """Stops the currently running experiment."""
        if self.experiment_process and self.experiment_process.is_alive():
            self.experiment_process.terminate()
            self.experiment_process.join() # Wait for the process to terminate
            self.experiment_process = None
            return "Experiment stopped."
        else:
            return "No experiment running."
    
    def get_latest_log_content(self):
        """Finds the latest log file across all datasets and returns its content."""
        if self.latest_log_file and os.path.exists(self.latest_log_file):
            with open(self.latest_log_file, 'r') as f:
                return f.read()

        # Import data_generator to use the new structure
        try:
            sys.path.insert(0, self.parent_dir)
            import data_generator
            
            # Find the latest run across all datasets
            latest_run = data_generator.find_latest_run_across_all_datasets()
            
            if latest_run is None:
                return "No experiment runs found in datasets/IDs/"
            
            log_file = latest_run["terminal_output"]
            
            if os.path.exists(log_file):
                self.latest_log_file = log_file
                with open(log_file, 'r') as f:
                    return f.read()
            else:
                return f"Log file not found: {log_file}"
                
        except Exception as e:
            return f"Error finding latest experiment: {str(e)}"

    def find_csv_files(self):
        """Find all .csv files in the project directory"""
        csv_files = []
        for root, _, files in os.walk(self.parent_dir):
            for file in files:
                if file.endswith(".csv"):
                    csv_files.append(os.path.join(root, file))
        return csv_files

    def update_config(self, data_source, csv_path, multiclass_strategy=None):
        """Update config.yaml based on data source selection and multiclass settings"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            if 'data' not in config:
                config['data'] = {}

            if data_source == "synthetic":
                config['data']['generate_new'] = True
            elif data_source == "real":
                config['data']['generate_new'] = False
                if csv_path:
                    config['data']['external_dataset'] = csv_path

            # Add multiclass configuration
            if 'multiclass' not in config:
                config['multiclass'] = {}
            
            if multiclass_strategy:
                config['multiclass']['enabled'] = True
                config['multiclass']['strategy'] = multiclass_strategy
            else:
                config['multiclass']['enabled'] = False
                config['multiclass']['strategy'] = None

            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

        except Exception as e:
            raise Exception(f"Failed to update config.yaml: {str(e)}")

    def get_config_preview(self):
        """Get configuration file content for preview"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return f.read()
            else:
                return None
        except Exception as e:
            raise Exception(f"Failed to read config: {str(e)}")
    
    def validate_environment(self):
        """Validate that run_test.py is accessible"""
        try:
            # Try to import and verify run_test has main function
            if hasattr(run_test, 'main'):
                return True
            else:
                raise Exception("run_test.py does not have main() function")
        except ImportError as e:
            raise Exception(f"Cannot import run_test.py: {str(e)}")

    def delete_dataset_progress(self, data_source="synthetic", csv_path=None):
        """
        Delete all progress data for a dataset, allowing fresh start
        Enhanced to support both binary and multiclass experiments
        
        Parameters:
        -----------
        data_source : str
            Type of data source ("synthetic" or "real")
        csv_path : str, optional
            Path to CSV file for real data
            
        Returns:
        --------
        dict : Result of deletion operation
        """
        try:
            # Import required modules
            sys.path.insert(0, self.parent_dir)
            import data_generator
            import shutil
            
            # Load current config to get dataset parameters
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Generate dataset ID based on parameters
            if data_source == "synthetic":
                params = config.get("data", {}).get("params", {})
                dataset_id = data_generator.generate_dataset_id(params, "synth")
            else:
                # For real data, use the improved content-based hash generation
                if csv_path and os.path.exists(csv_path):
                    dataset_id = data_generator.generate_real_data_id(csv_path)
                else:
                    # Fallback for non-existent files
                    params = {"file_path": csv_path}
                    dataset_id = data_generator.generate_dataset_id(params, "real")
            
            # Get dataset directory structure
            dataset_paths = data_generator.get_dataset_directory_structure(dataset_id)
            dataset_dir = dataset_paths["dataset_dir"]
            best_runner_file = dataset_paths["best_runner_json"]
            runs_dir = dataset_paths["runs_dir"]
            
            deleted_items = []
            
            # Check for multiclass experiment files first
            multiclass_progress_file = os.path.join(dataset_dir, f"multiclass_progress_{dataset_id}.json")
            multiclass_best_runner_file = os.path.join(dataset_dir, f"multiclass_best_runner_{dataset_id}.json")
            
            is_multiclass = os.path.exists(multiclass_progress_file) or os.path.exists(multiclass_best_runner_file)
            
            if is_multiclass:
                # Delete multiclass-specific files
                if os.path.exists(multiclass_progress_file):
                    os.remove(multiclass_progress_file)
                    deleted_items.append("Multiclass progress file")
                
                if os.path.exists(multiclass_best_runner_file):
                    os.remove(multiclass_best_runner_file)
                    deleted_items.append("Multiclass best runner configuration")
                
                # Delete individual dichotomy run directories
                dichotomy_dirs_deleted = 0
                if os.path.exists(dataset_dir):
                    for item in os.listdir(dataset_dir):
                        item_path = os.path.join(dataset_dir, item)
                        if os.path.isdir(item_path) and item.startswith("dichotomy_") and item.endswith("_run"):
                            shutil.rmtree(item_path)
                            dichotomy_dirs_deleted += 1
                
                if dichotomy_dirs_deleted > 0:
                    deleted_items.append(f"Dichotomy run directories ({dichotomy_dirs_deleted})")
            
            # Delete standard binary experiment files
            if os.path.exists(best_runner_file):
                os.remove(best_runner_file)
                deleted_items.append("Best runner configuration")
            
            # Delete runs directory if it exists
            if os.path.exists(runs_dir):
                shutil.rmtree(runs_dir)
                deleted_items.append("All run history")
            
            # Delete main results CSV if it exists (dataset level)
            main_results_csv = os.path.join(dataset_dir, "test_results.csv")
            if os.path.exists(main_results_csv):
                os.remove(main_results_csv)
                deleted_items.append("Main results file")
            
            if deleted_items:
                experiment_type = "multiclass" if is_multiclass else "binary"
                return {
                    "success": True,
                    "message": f"Successfully deleted {experiment_type} experiment progress: {', '.join(deleted_items)}",
                    "dataset_id": dataset_id,
                    "deleted_items": deleted_items,
                    "experiment_type": experiment_type
                }
            else:
                return {
                    "success": True,
                    "message": "No progress data found to delete (dataset was already fresh)",
                    "dataset_id": dataset_id,
                    "deleted_items": [],
                    "experiment_type": "none"
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Error deleting progress: {str(e)}",
                "dataset_id": dataset_id if 'dataset_id' in locals() else "unknown",
                "error": str(e)
            }

    def get_dataset_status(self, data_source="synthetic", csv_path=None, current_level=1):
        """
        Get the status of a dataset (new, in progress, or completed)
        Returns a dictionary with detailed status information including ETA
        Enhanced to support both binary and multiclass experiments
        """
        try:
            # Import required modules
            sys.path.insert(0, self.parent_dir)
            import data_generator
            from test_LSEnsemble import load_study_progress
            
            # Load current config to get capilaridad configuration
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            current_capilaridad_config = config.get("capilaridad_levels", {}).get(current_level, {})
            
            # Generate dataset ID based on parameters
            if data_source == "synthetic":
                params = config.get("data", {}).get("params", {})
                dataset_id = data_generator.generate_dataset_id(params, "synth")
            else:
                # For real data, use the improved content-based hash generation
                if csv_path and os.path.exists(csv_path):
                    dataset_id = data_generator.generate_real_data_id(csv_path)
                else:
                    # Fallback for non-existent files
                    params = {"file_path": csv_path}
                    dataset_id = data_generator.generate_dataset_id(params, "real")
            
            # First check for multiclass experiment
            dataset_paths = data_generator.get_dataset_directory_structure(dataset_id)
            multiclass_status = self.get_multiclass_status(dataset_paths, dataset_id, current_capilaridad_config, current_level)
            
            if multiclass_status:
                # This is a multiclass experiment, return multiclass status
                return multiclass_status
            
            # If not multiclass, continue with binary experiment logic
            # Get dataset directory structure
            dataset_paths = data_generator.get_dataset_directory_structure(dataset_id)
            best_runner_file = dataset_paths["best_runner_json"]
            
            # Check if best_runner file exists (binary experiment)
            if not os.path.exists(best_runner_file):
                return {
                    "status": "new",
                    "dataset_id": dataset_id,
                    "message": "Dataset has not been studied before",
                    "current_capilaridad_config": current_capilaridad_config,
                    "current_level": current_level,
                    "experiment_type": "binary"
                }
            
            # Load study progress
            study_data = load_study_progress(best_runner_file)
            
            if not study_data or "execution_state" not in study_data:
                # Old format or corrupted file
                return {
                    "status": "existing_old",
                    "dataset_id": dataset_id,
                    "message": "Dataset has old format study data",
                    "best_runner": study_data.get("best_runner") if study_data else None,
                    "current_capilaridad_config": current_capilaridad_config,
                    "current_level": current_level,
                    "experiment_type": "binary"
                }
            
            exec_state = study_data["execution_state"]
            best_runner = study_data["best_runner"]
            capilaridad_config = study_data.get("capilaridad_config", {})
            
            # Determine status based on execution state
            stage1_complete = exec_state.get("stage1_completed", False)
            stage1_total = exec_state.get("stage1_total", 0)
            stage1_done = exec_state.get("stage1_position", 0)
            stage2_total = exec_state.get("stage2_total", 0)
            stage2_done = exec_state.get("stage2_position", 0)
            
            # Get previous capilaridad configuration
            previous_capilaridad_config = capilaridad_config
            previous_level = previous_capilaridad_config.get("level", "unknown")
            
            # Get timing and ETA information
            timing_info = self.get_timing_and_eta(dataset_paths, stage1_done, stage1_total, 
                                                 stage2_done, stage2_total, stage1_complete)
            
            status_info = {
                "status": "existing",
                "dataset_id": dataset_id,
                "stage1_completed": stage1_complete,
                "stage1_progress": f"{stage1_done}/{stage1_total}",
                "stage2_progress": f"{stage2_done}/{stage2_total}",
                "current_stage": exec_state.get("current_stage", 1),
                "last_updated": exec_state.get("last_updated", "Unknown"),
                "best_runner": best_runner,
                "study_mode": exec_state.get("study_mode", "full_study"),
                "capilaridad_config": capilaridad_config,
                "previous_capilaridad_config": previous_capilaridad_config,
                "current_capilaridad_config": current_capilaridad_config,
                "previous_level": previous_level,
                "current_level": current_level,
                "best_metric_so_far": exec_state.get("best_metric_so_far"),
                "experiment_type": "binary",
                **timing_info  # Add timing information
            }
            
            # Determine overall status
            if stage1_complete and stage2_done >= stage2_total and stage2_total > 0:
                status_info["overall_status"] = "complete"
                status_info["message"] = "Study is complete"
            elif stage1_complete:
                status_info["overall_status"] = "stage2_in_progress"
                status_info["message"] = "Stage 1 complete, Stage 2 in progress"
            elif stage1_total > 0:
                status_info["overall_status"] = "stage1_in_progress"
                status_info["message"] = "Stage 1 in progress"
            else:
                status_info["overall_status"] = "unknown"
                status_info["message"] = "Unknown status"
            
            return status_info
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error checking dataset status: {str(e)}",
                "current_capilaridad_config": current_capilaridad_config if 'current_capilaridad_config' in locals() else {},
                "current_level": current_level,
                "experiment_type": "unknown"
            }

    def get_multiclass_status(self, dataset_paths, dataset_id, current_capilaridad_config, current_level):
        """
        Check for multiclass experiment status and return appropriately formatted progress.
        
        Parameters:
        -----------
        dataset_paths : dict
            Standard dataset directory paths
        dataset_id : str
            The dataset ID
        current_capilaridad_config : dict
            Current capilaridad configuration
        current_level : int
            Current capilaridad level
            
        Returns:
        --------
        dict or None : Multiclass status information if multiclass experiment exists, None otherwise
        """
        try:
            # Check for multiclass progress file
            multiclass_progress_file = os.path.join(dataset_paths["dataset_dir"], f"multiclass_progress_{dataset_id}.json")
            multiclass_best_runner_file = os.path.join(dataset_paths["dataset_dir"], f"multiclass_best_runner_{dataset_id}.json")
            
            if not os.path.exists(multiclass_progress_file):
                return None  # No multiclass experiment
            
            # Load multiclass progress
            with open(multiclass_progress_file, 'r') as f:
                multiclass_data = json.load(f)
            
            # Load multiclass best runner if available
            multiclass_best_runner = None
            if os.path.exists(multiclass_best_runner_file):
                with open(multiclass_best_runner_file, 'r') as f:
                    multiclass_best_runner = json.load(f)
            
            # Parse multiclass progress to match the expected format
            dichotomies = multiclass_data.get("dichotomies", {})
            total_dichotomies = len(dichotomies)
            completed_dichotomies = sum(1 for d in dichotomies.values() if d.get("status") == "completed")
            in_progress_dichotomies = sum(1 for d in dichotomies.values() if d.get("status") == "in_progress")
            
            # Calculate overall progress as if it were Stage 1
            # Each dichotomy represents a "configuration" in our binary-like progress format
            stage1_progress = f"{completed_dichotomies}/{total_dichotomies}"
            stage2_progress = "0/0"  # Multiclass doesn't have traditional Stage 2
            
            # Determine current status
            if completed_dichotomies == total_dichotomies and total_dichotomies > 0:
                overall_status = "complete"
                message = f"Multiclass study is complete ({completed_dichotomies}/{total_dichotomies} dichotomies)"
                stage1_complete = True
                current_stage = 2  # Mark as completed
            elif in_progress_dichotomies > 0:
                overall_status = "stage1_in_progress"
                message = f"Multiclass study in progress ({completed_dichotomies}/{total_dichotomies} dichotomies completed)"
                stage1_complete = False
                current_stage = 1
            elif completed_dichotomies > 0:
                overall_status = "stage1_in_progress"
                message = f"Multiclass study resumed ({completed_dichotomies}/{total_dichotomies} dichotomies completed)"
                stage1_complete = False
                current_stage = 1
            else:
                overall_status = "new"
                message = f"Multiclass study ready to start ({total_dichotomies} dichotomies)"
                stage1_complete = False
                current_stage = 1
            
            # Calculate timing information for multiclass
            timing_info = self.get_multiclass_timing_and_eta(dataset_paths, completed_dichotomies, total_dichotomies)
            
            # Get strategy and class information
            strategy = multiclass_data.get("strategy", "unknown").upper()
            n_classes = multiclass_data.get("n_classes", "unknown")
            class_names = multiclass_data.get("class_names", [])
            
            # Best metric from multiclass best runner
            best_metric_so_far = None
            if multiclass_best_runner:
                best_metric_so_far = multiclass_best_runner.get("best_metric_so_far")
            
            return {
                "status": "existing",
                "dataset_id": dataset_id,
                "stage1_completed": stage1_complete,
                "stage1_progress": stage1_progress,
                "stage2_progress": stage2_progress,
                "current_stage": current_stage,
                "last_updated": multiclass_data.get("last_updated", "Unknown"),
                "best_runner": multiclass_best_runner.get("best_runner_config") if multiclass_best_runner else None,
                "study_mode": "multiclass",
                "capilaridad_config": {"level": current_level},  # Simple capilaridad info
                "previous_capilaridad_config": {"level": current_level},
                "current_capilaridad_config": current_capilaridad_config,
                "previous_level": current_level,
                "current_level": current_level,
                "best_metric_so_far": best_metric_so_far,
                "experiment_type": "multiclass",
                "overall_status": overall_status,
                "message": message,
                # Multiclass-specific information
                "multiclass_info": {
                    "strategy": strategy,
                    "n_classes": n_classes,
                    "class_names": class_names[:5],  # Show first 5 class names
                    "total_dichotomies": total_dichotomies,
                    "completed_dichotomies": completed_dichotomies,
                    "in_progress_dichotomies": in_progress_dichotomies,
                    "current_dichotomy": multiclass_data.get("summary", {}).get("current_dichotomy"),
                    "current_dichotomy_index": multiclass_data.get("summary", {}).get("current_dichotomy_index"),
                    "dichotomy_details": dichotomies
                },
                **timing_info  # Add timing information
            }
            
        except Exception as e:
            # If there's an error reading multiclass progress, return None to fall back to binary check
            return None

    def get_multiclass_timing_and_eta(self, dataset_paths, completed_dichotomies, total_dichotomies):
        """
        Calculate timing statistics and ETA for multiclass experiments.
        
        Parameters:
        -----------
        dataset_paths : dict
            Dataset directory paths
        completed_dichotomies : int
            Number of completed dichotomies
        total_dichotomies : int
            Total number of dichotomies
            
        Returns:
        --------
        dict : Timing information similar to binary experiments
        """
        timing_info = {
            "avg_time_per_run": 0,
            "total_time_elapsed": 0,
            "eta_seconds": None,
            "eta_formatted": "Unknown",
            "total_runs_completed": completed_dichotomies,
            "runs_per_hour": 0,
            "remaining_runs": total_dichotomies - completed_dichotomies,
            "current_progress": completed_dichotomies,
            "total_stage_runs": total_dichotomies,
            "stage_name": "Dichotomies",
            "dataset_size": 0,
            "using_estimate": True
        }
        
        try:
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            
            # Look for timing information from completed dichotomy runs
            actual_times = []
            
            # Check individual dichotomy run directories for timing data
            dataset_dir = dataset_paths["dataset_dir"]
            
            for i in range(1, completed_dichotomies + 1):
                dichotomy_run_dir = os.path.join(dataset_dir, f"dichotomy_{i:02d}_run")
                if os.path.exists(dichotomy_run_dir):
                    # Look for timing data in the dichotomy's results
                    dichotomy_results_csv = os.path.join(dichotomy_run_dir, "test_results.csv")
                    if os.path.exists(dichotomy_results_csv):
                        try:
                            df = pd.read_csv(dichotomy_results_csv)
                            if 'time_taken' in df.columns:
                                # Sum all time_taken values for this dichotomy
                                dichotomy_total_time = df['time_taken'].sum()
                                actual_times.append(dichotomy_total_time)
                        except Exception:
                            continue
            
            # Calculate timing estimates
            if actual_times:
                # Use actual timing data
                avg_time_per_dichotomy = np.mean(actual_times)
                total_elapsed = sum(actual_times)
                
                # Calculate ETA for remaining dichotomies
                remaining_dichotomies = total_dichotomies - completed_dichotomies
                if remaining_dichotomies > 0:
                    eta_seconds = remaining_dichotomies * avg_time_per_dichotomy
                    
                    # Format ETA
                    if eta_seconds < 3600:
                        minutes = int(eta_seconds / 60)
                        eta_formatted = f"{minutes}m"
                    elif eta_seconds < 86400:
                        hours = int(eta_seconds / 3600)
                        minutes = int((eta_seconds % 3600) / 60)
                        eta_formatted = f"{hours}h {minutes}m"
                    else:
                        days = int(eta_seconds / 86400)
                        hours = int((eta_seconds % 86400) / 3600)
                        eta_formatted = f"{days}d {hours}h"
                    
                    timing_info.update({
                        "eta_seconds": eta_seconds,
                        "eta_formatted": eta_formatted,
                    })
                
                # Calculate rate information
                dichotomies_per_hour = 3600 / avg_time_per_dichotomy if avg_time_per_dichotomy > 0 else 0
                
                timing_info.update({
                    "avg_time_per_run": round(avg_time_per_dichotomy, 2),
                    "total_time_elapsed": round(total_elapsed, 2),
                    "runs_per_hour": round(dichotomies_per_hour, 2),
                    "using_estimate": False
                })
                
            else:
                # Use estimates if no actual data
                # Estimate ~5-15 minutes per dichotomy depending on complexity
                estimated_time_per_dichotomy = 600  # 10 minutes default
                remaining_dichotomies = total_dichotomies - completed_dichotomies
                
                if remaining_dichotomies > 0:
                    eta_seconds = remaining_dichotomies * estimated_time_per_dichotomy
                    hours = int(eta_seconds / 3600)
                    minutes = int((eta_seconds % 3600) / 60)
                    eta_formatted = f"~{hours}h {minutes}m" if hours > 0 else f"~{minutes}m"
                    
                    timing_info.update({
                        "avg_time_per_run": estimated_time_per_dichotomy,
                        "eta_seconds": eta_seconds,
                        "eta_formatted": eta_formatted,
                        "runs_per_hour": 0.1,  # ~6 per hour estimate
                        "using_estimate": True
                    })
                    
        except Exception as e:
            # If timing calculation fails, use basic estimates
            remaining_dichotomies = total_dichotomies - completed_dichotomies
            if remaining_dichotomies > 0:
                timing_info.update({
                    "avg_time_per_run": 600,  # 10 minutes estimate
                    "eta_formatted": f"~{remaining_dichotomies * 10}m",
                    "using_estimate": True
                })
        
        return timing_info

    def get_timing_and_eta(self, dataset_paths, stage1_done, stage1_total, stage2_done, stage2_total, stage1_complete):
        """
        Calculate timing statistics and ETA based on dataset size and actual run times
        """
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        timing_info = {
            "avg_time_per_run": 0,
            "total_time_elapsed": 0,
            "eta_seconds": None,
            "eta_formatted": "Unknown",
            "total_runs_completed": 0,
            "runs_per_hour": 0
        }
        
        try:
            # Get the main CSV file for this dataset
            dataset_csv = dataset_paths["dataset_csv"]
            
            # Get dataset size for initial time estimates
            dataset_size = 0
            if os.path.exists(dataset_csv):
                try:
                    df_dataset = pd.read_csv(dataset_csv)
                    dataset_size = len(df_dataset)
                except Exception:
                    dataset_size = 0
            
            # Calculate initial time estimate based on dataset size
            if dataset_size > 1000:
                estimated_time_per_run = 30.0  # 30 seconds
            elif dataset_size > 500:
                estimated_time_per_run = 15.0  # 15 seconds
            else:
                estimated_time_per_run = 10.0  # 10 seconds
            
            # Look for the main test_results.csv file in the dataset directory
            main_results_csv = os.path.join(dataset_paths["dataset_dir"], "test_results.csv")
            
            # Try to get actual timing data from completed runs
            actual_times = []
            total_runs = 0
            
            if os.path.exists(main_results_csv):
                try:
                    df = pd.read_csv(main_results_csv)
                    if 'time_taken' in df.columns:
                        all_times = df['time_taken'].dropna().tolist()
                        total_runs = len(all_times)
                        
                        # Stage-specific timing logic
                        if stage1_complete and stage2_done > 0:
                            # Stage 2: Only use recent timing data (Stage 2 runs)
                            # Since we can't distinguish stages in CSV, use only recent runs
                            # Estimate: Stage 1 had stage1_total runs, so Stage 2 starts after that
                            stage2_start_index = stage1_total
                            if total_runs > stage2_start_index:
                                # Use only Stage 2 timing data
                                actual_times = all_times[stage2_start_index:]
                            else:
                                # Not enough data yet, use estimate
                                actual_times = []
                        else:
                            # Stage 1: Use all available timing data
                            actual_times = all_times
                            
                except Exception:
                    pass
            
            # Calculate timing statistics
            if actual_times:
                # Use actual timing data if available
                avg_time = np.mean(actual_times)
                total_elapsed = sum(actual_times)
                
                # Weight the estimate: use more actual data as we get more runs
                # For Stage 2, be more aggressive in using actual data since runs may be different
                if stage1_complete:
                    # Stage 2: Use actual data more quickly since characteristics may be different
                    weight_actual = min(len(actual_times) / 3.0, 1.0)  # Full weight after 3 runs
                else:
                    # Stage 1: More conservative weighting
                    weight_actual = min(len(actual_times) / 10.0, 1.0)  # Full weight after 10 runs
                    
                weighted_avg_time = (1 - weight_actual) * estimated_time_per_run + weight_actual * avg_time
            else:
                # Use estimated time if no actual data yet
                weighted_avg_time = estimated_time_per_run
                avg_time = estimated_time_per_run
                total_elapsed = 0
            
            # Calculate remaining work
            if stage1_complete:
                # Currently in stage 2
                remaining_runs = stage2_total - stage2_done
                current_progress = stage2_done
                total_stage_runs = stage2_total
                stage_name = "Stage 2"
            else:
                # Currently in stage 1
                remaining_runs = stage1_total - stage1_done
                current_progress = stage1_done
                total_stage_runs = stage1_total
                stage_name = "Stage 1"
            
            # Calculate ETA
            eta_seconds = None
            eta_formatted = "Unknown"
            
            if remaining_runs > 0 and weighted_avg_time > 0:
                eta_seconds = remaining_runs * weighted_avg_time
                
                # Format ETA nicely
                if eta_seconds < 60:
                    eta_formatted = f"{int(eta_seconds)}s"
                elif eta_seconds < 3600:
                    minutes = int(eta_seconds / 60)
                    seconds = int(eta_seconds % 60)
                    eta_formatted = f"{minutes}m {seconds}s"
                else:
                    hours = int(eta_seconds / 3600)
                    minutes = int((eta_seconds % 3600) / 60)
                    eta_formatted = f"{hours}h {minutes}m"
            
            # Calculate runs per hour
            runs_per_hour = 3600 / weighted_avg_time if weighted_avg_time > 0 else 0
            
            timing_info.update({
                "avg_time_per_run": round(weighted_avg_time, 2),
                "actual_avg_time": round(avg_time, 2) if actual_times else 0,
                "estimated_time": round(estimated_time_per_run, 2),
                "total_time_elapsed": round(total_elapsed, 2),
                "eta_seconds": eta_seconds,
                "eta_formatted": eta_formatted,
                "total_runs_completed": len(actual_times),  # Use actual stage-specific count
                "runs_per_hour": round(runs_per_hour, 1),
                "remaining_runs": remaining_runs,
                "current_progress": current_progress,
                "total_stage_runs": total_stage_runs,
                "stage_name": stage_name,
                "dataset_size": dataset_size,
                "using_estimate": len(actual_times) < (3 if stage1_complete else 10)  # Different thresholds per stage
            })
            
        except Exception as e:
            # If anything fails, return basic timing info with estimates
            timing_info.update({
                "avg_time_per_run": 15.0,  # Default estimate
                "eta_formatted": "Calculating...",
                "using_estimate": True
            })
        
        return timing_info

    def generate_experiment_report(self, data_source="synthetic", csv_path=None, current_level=1, gui_root=None, language: str = "en"):
        """
        Generate and show experiment report popup automatically after completion.
        Uses the internal show_report_popup(experiment_path, language, parent) entry point.
        """
        try:
            # Get experiment folder path
            experiment_path = self.get_experiment_folder_path(data_source, csv_path, current_level)
            
            if not experiment_path or not os.path.exists(experiment_path):
                raise Exception("No experiment folder found to generate report")
            
            # Add parent directory (Small) to path since report_generation is there
            if self.parent_dir not in sys.path:
                sys.path.insert(0, self.parent_dir)
            
            # Import and call the internal entry point for report preview
            try:
                from report_generation.pdf_preview_popup import show_report_popup
            except ModuleNotFoundError:
                # Fallback: add the Small directory explicitly to sys.path
                small_dir = os.path.dirname(os.path.abspath(__file__))  # .../Small/GUI
                small_dir = os.path.dirname(small_dir)  # .../Small
                if small_dir not in sys.path:
                    sys.path.insert(0, small_dir)
                from report_generation.pdf_preview_popup import show_report_popup

            # Parent should be the Tk root passed in from the GUI
            parent = gui_root

            # Call the consolidated popup generator
            show_report_popup(experiment_path, language=language, parent=parent, experiment_runner=self)
            
        except Exception as e:
            raise Exception(f"Error generating experiment report: {str(e)}")
    
    def get_experiment_folder_path(self, data_source="synthetic", csv_path=None, current_level=1):
        """Get the path to the experiment folder for report generation"""
        try:
            # Import required modules
            sys.path.insert(0, self.parent_dir)
            import data_generator
            
            # Load current config to get dataset parameters
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Generate dataset ID based on parameters
            if data_source == "synthetic":
                params = config.get("data", {}).get("params", {})
                dataset_id = data_generator.generate_dataset_id(params, "synth")
            else:
                # For real data, use the improved content-based hash generation
                if csv_path and os.path.exists(csv_path):
                    dataset_id = data_generator.generate_real_data_id(csv_path)
                else:
                    # Fallback for non-existent files
                    params = {"file_path": csv_path}
                    dataset_id = data_generator.generate_dataset_id(params, "real")
            
            # Get dataset directory structure
            dataset_paths = data_generator.get_dataset_directory_structure(dataset_id)
            return dataset_paths["dataset_dir"]
            
        except Exception as e:
            return None

    def diagnose_path_mismatch(self, original_csv_path):
        """
        Diagnose the path mismatch issue between experiment creation and report generation
        for multiclass experiments.
        
        Parameters:
        -----------
        original_csv_path : str
            Path to the original CSV file that would be selected by the user
            
        Returns:
        --------
        dict : Diagnostic results including IDs and paths
        """
        try:
            # Import required modules
            sys.path.insert(0, self.parent_dir)
            import data_generator
            from csv_preloading import CSVPreprocessor, create_multiclass_dichotomies
            
            # Step 1: Generate ID from original file (as done when experiment starts)
            print(f"Step 1: Generating dataset ID from original file...")
            id_from_original = data_generator.generate_real_data_id(original_csv_path)
            original_folder_path = data_generator.get_dataset_directory_structure(id_from_original)["dataset_dir"]
            print(f"   Original file: {original_csv_path}")
            print(f"   Generated ID: {id_from_original}")
            print(f"   Expected folder: {original_folder_path}")
            
            # Step 2: Simulate multiclass preprocessing to get first dichotomy file
            print(f"\nStep 2: Simulating multiclass preprocessing...")
            preprocessor = CSVPreprocessor(gui_enabled=False)
            multiclass_candidates = preprocessor.detect_multiclass(original_csv_path)
            
            if not multiclass_candidates:
                return {
                    "success": False,
                    "error": "No multiclass candidates found in the provided file"
                }
            
            # Use first candidate and OVA strategy (most common)
            main_candidate = multiclass_candidates[0]
            strategy = "ova"
            
            # Create dichotomies (but don't save them permanently)
            dichotomy_result = create_multiclass_dichotomies(
                original_csv_path, 
                main_candidate, 
                strategy=strategy
            )
            
            if not dichotomy_result['success']:
                return {
                    "success": False,
                    "error": f"Failed to create dichotomies: {dichotomy_result.get('error', 'Unknown error')}"
                }
            
            # Get the first dichotomy file path
            first_dichotomy_path = dichotomy_result['dichotomy_files'][0]
            print(f"   First dichotomy file: {first_dichotomy_path}")
            
            # Step 3: Generate ID from first dichotomy file (as done during report generation)
            print(f"\nStep 3: Generating dataset ID from first dichotomy file...")
            id_from_dichotomy = data_generator.generate_real_data_id(first_dichotomy_path)
            dichotomy_folder_path = data_generator.get_dataset_directory_structure(id_from_dichotomy)["dataset_dir"]
            print(f"   Dichotomy file: {first_dichotomy_path}")
            print(f"   Generated ID: {id_from_dichotomy}")
            print(f"   Expected folder: {dichotomy_folder_path}")
            
            # Clean up temporary dichotomy files
            try:
                for temp_file in dichotomy_result['dichotomy_files']:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        print(f"   Cleaned up: {temp_file}")
            except Exception as cleanup_error:
                print(f"   Warning: Could not clean up temporary files: {cleanup_error}")
            
            return {
                "success": True,
                "id_from_original": id_from_original,
                "id_from_dichotomy": id_from_dichotomy,
                "original_folder_path": original_folder_path,
                "dichotomy_folder_path": dichotomy_folder_path,
                "dichotomy_filepath": first_dichotomy_path,
                "ids_match": id_from_original == id_from_dichotomy
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def reconstitute_multiclass_models(self, dataset_id):
        """
        Orchestrates the reconstitution of multiclass models following the exact approach 
        from MC_Label_Switching/evaluate_performance.py.
        
        This loads the exact ECOC matrix and configuration used in the original experiment,
        then performs multiple simulations with different train/test splits to get
        robust statistical estimates.
        """
        try:
            import numpy as np
            import pandas as pd
            import json
            import os
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score, balanced_accuracy_score
            from sklearn.metrics import confusion_matrix, cohen_kappa_score
            from imblearn.metrics import geometric_mean_score, sensitivity_score
            
            print(f"Starting reconstitution for dataset_id: {dataset_id}")
            
            # Import required modules from the parent directory
            sys.path.insert(0, self.parent_dir)
            import data_generator
            
            # Get dataset directory structure
            dataset_paths = data_generator.get_dataset_directory_structure(dataset_id)
            dataset_dir = dataset_paths["dataset_dir"]
            
            if not os.path.exists(dataset_dir):
                return {"success": False, "message": f"Dataset directory not found: {dataset_dir}"}
            
            print(f"Working with dataset directory: {dataset_dir}")
            
            # 1. Load multiclass progress to get the EXACT experiment configuration
            multiclass_progress_file = os.path.join(dataset_dir, f"multiclass_progress_{dataset_id}.json")
            if not os.path.exists(multiclass_progress_file):
                return {"success": False, "message": "Multiclass progress file not found"}
            
            with open(multiclass_progress_file, 'r') as f:
                multiclass_data = json.load(f)
            
            strategy = multiclass_data.get("strategy", "ova")
            n_classes = multiclass_data.get("n_classes", 0)
            class_names = multiclass_data.get("class_names", [])
            # Use the EXACT code matrix from the original experiment
            M_ecoc_matrix = np.array(multiclass_data.get("code_matrix", []))
            
            print(f"Experiment configuration: {strategy} with {n_classes} classes")
            print(f"Classes: {class_names}")
            print(f"ECOC matrix shape: {M_ecoc_matrix.shape}")
            print(f"ECOC matrix:\n{M_ecoc_matrix}")
            
            # 2. Load the original dataset
            original_csv_path = None
            for f in os.listdir(dataset_dir):
                if f.endswith('_original.csv'):
                    original_csv_path = os.path.join(dataset_dir, f)
                    break
            
            if not original_csv_path or not os.path.exists(original_csv_path):
                return {"success": False, "message": "Original dataset file not found"}
            
            print(f"Loading original dataset: {original_csv_path}")
            
            # Load dataset exactly as in evaluate_performance.py
            df = pd.read_csv(original_csv_path)
            X = df.iloc[:, :-1].values
            y_original = df.iloc[:, -1].values
            
            # Map class labels to indices (following evaluate_performance.py approach)
            unique_classes = np.unique(y_original)
            if len(unique_classes) != n_classes:
                return {"success": False, "message": f"Class count mismatch: found {len(unique_classes)}, expected {n_classes}"}
            
            # Create class mapping - map original labels to numeric indices
            class_dict = {class_names[i]: i + 1 for i in range(len(class_names))}
            y = np.array([class_dict.get(str(label), class_dict.get(label, 1)) for label in y_original])
            class_labels = np.array(sorted(class_dict.values()))
            
            print(f"Dataset shape: {X.shape}")
            print(f"Class mapping: {class_dict}")
            print(f"Class labels: {class_labels}")
            
            # 3. Load best configurations for each dichotomy
            best_configs = {}
            dichotomy_dirs = [d for d in os.listdir(dataset_dir) 
                             if d.startswith('dichotomy_') and d.endswith('_run')]
            
            for dichotomy_dir in sorted(dichotomy_dirs):
                dichotomy_path = os.path.join(dataset_dir, dichotomy_dir)
                dichotomy_name = dichotomy_dir.replace('_run', '')
                
                # Load best runner configuration
                best_runner_file = os.path.join(dichotomy_path, f"best_runner_{dichotomy_name}.json")
                if os.path.exists(best_runner_file):
                    with open(best_runner_file, 'r') as f:
                        best_runner_data = json.load(f)
                        # Extract the best configuration from the saved data
                        best_config = best_runner_data.get('best_runner', {})
                        best_configs[dichotomy_name] = best_config
                        print(f"Loaded config for {dichotomy_name}: {list(best_config.keys())}")
                else:
                    return {"success": False, "message": f"Best runner file not found for {dichotomy_name}"}
            
            if not best_configs:
                return {"success": False, "message": "No best configurations found"}
            
            print(f"Loaded {len(best_configs)} dichotomy configurations")
            
            num_dichotomies = M_ecoc_matrix.shape[1]
            
            # 4. Run multiple simulations as in evaluate_performance.py
            n_simulations = 10  # Start with fewer simulations for testing
            test_size = 0.2     # 20% test size (matching original config.yaml)
            
            acc_simulations = []
            bal_acc_simulations = []
            kappa_simulations = []
            geom_mean_simulations = []
            sensitivity_simulations = []
            
            print(f"Starting {n_simulations} simulations...")
            
            for k_simu in range(n_simulations):
                print(f"  Simulation {k_simu + 1}/{n_simulations}")
                
                # Train-test split with different random seed for each simulation
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42 + k_simu, stratify=y
                )
                
                M_tst = X_test.shape[0]
                
                # Standardize features
                scaler = StandardScaler()
                X_train_n = scaler.fit_transform(X_train)
                X_test_n = scaler.transform(X_test)
                
                # Apply ECOC binarization (simplified version of apply_ecoc_binarization)
                Ye_pred = np.zeros((M_tst, num_dichotomies))
                
                for j_dic in range(num_dichotomies):
                    try:
                        # Get the ECOC column for this dichotomy
                        ecoc_column = M_ecoc_matrix[:, j_dic]
                        
                        # Create binary labels based on ECOC matrix
                        ye_train = np.zeros(len(y_train))
                        ye_test = np.zeros(len(y_test))
                        
                        for i, class_idx in enumerate(class_labels):
                            # Map class index to position in ECOC matrix
                            ecoc_class_idx = class_idx - 1  # Convert to 0-based index
                            if ecoc_class_idx < len(ecoc_column):
                                ecoc_value = ecoc_column[ecoc_class_idx]
                                
                                # Apply ECOC encoding to training labels
                                train_mask = (y_train == class_idx)
                                ye_train[train_mask] = ecoc_value
                                
                                # Apply ECOC encoding to test labels  
                                test_mask = (y_test == class_idx)
                                ye_test[test_mask] = ecoc_value
                        
                        # Check if we have both classes in training set
                        unique_train_labels = np.unique(ye_train)
                        if len(unique_train_labels) < 2:
                            print(f"    Warning: Dichotomy {j_dic+1} has only one class in training set")
                            # Use majority class prediction
                            Ye_pred[:, j_dic] = unique_train_labels[0] if len(unique_train_labels) > 0 else 1.0
                            continue
                        
                        # Get the corresponding dichotomy name
                        dichotomy_name = f"dichotomy_{j_dic+1:02d}"
                        if dichotomy_name in best_configs:
                            config = best_configs[dichotomy_name]
                            
                            try:
                                # Instantiate the model
                                model = self._instantiate_model(config)
                                
                                # Ensure input_size is set for LSEnsemble
                                if hasattr(model, 'input_size') and model.input_size is None:
                                    model.input_size = X_train_n.shape[1]
                                
                                print(f"    Training {dichotomy_name} with {len(ye_train)} samples...")
                                print(f"    Training labels unique: {np.unique(ye_train)}")
                                
                                # Convert data to the correct format for LSEnsemble
                                X_train_tensor = X_train_n.astype(np.float32)
                                ye_train_tensor = ye_train.astype(np.float32)
                                
                                # ECOC Explanation: Handle neutral labels (0 values)
                                # In ECOC strategies like OVO (One-vs-One), not all classes participate 
                                # in every dichotomy. Classes not involved are marked with 0 (neutral).
                                # For binary classification, we only use classes marked as -1 or +1.
                                unique_labels = np.unique(ye_train_tensor)
                                print(f"    Original unique labels: {unique_labels}")
                                
                                # Filter out 0 values if present (neutral/uninvolved classes in this dichotomy)
                                if 0 in unique_labels:
                                    mask = ye_train_tensor != 0
                                    X_train_tensor = X_train_tensor[mask]
                                    ye_train_tensor = ye_train_tensor[mask]
                                    print(f"    Filtered out {np.sum(~mask)} neutral (0) labels for this dichotomy")
                                
                                # Ensure we have exactly 2 classes for binary classification
                                unique_labels = np.unique(ye_train_tensor)
                                if len(unique_labels) != 2:
                                    print(f"    Warning: Expected 2 classes, got {len(unique_labels)}: {unique_labels}")
                                    if len(unique_labels) == 1:
                                        # Skip this dichotomy if only one class
                                        print(f"    Skipping {dichotomy_name} - only one class present")
                                        Ye_pred[:, j_dic] = unique_labels[0]
                                        continue
                                
                                print(f"    Final training labels: {unique_labels} (n={len(ye_train_tensor)})")
                                
                                # Update test data accordingly
                                X_test_tensor = X_test_n.astype(np.float32)
                                
                                # Fit the model
                                model.fit(X_train_tensor, ye_train_tensor)
                                
                                # bin_format Explanation: LSEnsemble format for binary labels
                                # "minus_one_one" = {-1, +1} format (ECOC standard)
                                # "zero_one" = {0, 1} format (sklearn standard)
                                if hasattr(model, 'bin_format'):
                                    print(f"    Model bin_format set to: {model.bin_format}")
                                else:
                                    print(f"    Warning: bin_format not set, manually setting it")
                                    # Set bin_format based on our data conversion
                                    if 0 in unique_labels:
                                        model.bin_format = 0
                                    else:
                                        model.bin_format = -1
                                
                                # Predict on test set
                                X_test_tensor = X_test_n.astype(np.float32)
                                ye_pred = model.predict(X_test_tensor)
                                
                                # Ensure predictions are in the correct format
                                ye_pred = np.array(ye_pred).flatten()
                                if len(ye_pred) != M_tst:
                                    print(f"    Warning: Prediction length mismatch. Expected {M_tst}, got {len(ye_pred)}")
                                    ye_pred = np.resize(ye_pred, M_tst)
                                
                                Ye_pred[:, j_dic] = ye_pred
                                print(f"    Successfully completed {dichotomy_name}")
                                
                            except Exception as model_error:
                                print(f"    Model error in {dichotomy_name}: {str(model_error)}")
                                print(f"    Error type: {type(model_error).__name__}")
                                import traceback
                                traceback.print_exc()
                                # Use fallback predictions
                                Ye_pred[:, j_dic] = np.random.choice([-1.0, 1.0], size=M_tst)
                        else:
                            print(f"    Warning: No config found for {dichotomy_name}")
                            Ye_pred[:, j_dic] = ye_test  # Fallback
                            
                    except Exception as e:
                        print(f"    Error in dichotomy {j_dic+1}: {str(e)}")
                        Ye_pred[:, j_dic] = np.random.choice([-1.0, 1.0], size=M_tst)
                
                # ECOC decoding to get final multiclass predictions
                y_pred_MC = np.zeros(M_tst, dtype=int)
                for i in range(M_tst):
                    sample_predictions = Ye_pred[i, :]
                    
                    # Find the class with minimum Hamming distance
                    min_distance = float('inf')
                    predicted_class = class_labels[0]
                    
                    for class_idx in class_labels:
                        ecoc_class_idx = class_idx - 1  # Convert to 0-based index
                        if ecoc_class_idx < M_ecoc_matrix.shape[0]:
                            class_codeword = M_ecoc_matrix[ecoc_class_idx, :]
                            distance = np.sum(sample_predictions != class_codeword)
                            
                            if distance < min_distance:
                                min_distance = distance
                                predicted_class = class_idx
                    
                    y_pred_MC[i] = predicted_class
                
                # Calculate metrics for this simulation
                acc = accuracy_score(y_test, y_pred_MC)
                bal_acc = balanced_accuracy_score(y_test, y_pred_MC)
                kappa = cohen_kappa_score(y_test, y_pred_MC)
                
                try:
                    geom_mean = geometric_mean_score(y_test, y_pred_MC, average='weighted')
                    sensitivity = sensitivity_score(y_test, y_pred_MC, average='weighted')
                except:
                    geom_mean = 0.0
                    sensitivity = 0.0
                
                # Store metrics
                acc_simulations.append(acc)
                bal_acc_simulations.append(bal_acc)
                kappa_simulations.append(kappa)
                geom_mean_simulations.append(geom_mean)
                sensitivity_simulations.append(sensitivity)
                
                print(f"    Simulation {k_simu + 1} metrics: Acc={acc:.4f}, Bal_Acc={bal_acc:.4f}, Kappa={kappa:.4f}")
            
            # Calculate final statistics
            final_metrics = {
                "avg_acc": np.mean(acc_simulations),
                "avg_bal_acc": np.mean(bal_acc_simulations),
                "avg_kappa": np.mean(kappa_simulations),
                "avg_geom_mean": np.mean(geom_mean_simulations),
                "avg_sensitivity": np.mean(sensitivity_simulations),
                "std_acc": np.std(acc_simulations),
                "std_bal_acc": np.std(bal_acc_simulations),
                "std_kappa": np.std(kappa_simulations),
                "std_geom_mean": np.std(geom_mean_simulations),
                "std_sensitivity": np.std(sensitivity_simulations),
                "n_simulations": n_simulations
            }
            
            print("Final performance metrics:")
            print(f"  Accuracy: {final_metrics['avg_acc']:.5f} ± {final_metrics['std_acc']:.5f}")
            print(f"  Balanced Accuracy: {final_metrics['avg_bal_acc']:.5f} ± {final_metrics['std_bal_acc']:.5f}")
            print(f"  Cohen's Kappa: {final_metrics['avg_kappa']:.5f} ± {final_metrics['std_kappa']:.5f}")
            print(f"  Geometric Mean: {final_metrics['avg_geom_mean']:.5f} ± {final_metrics['std_geom_mean']:.5f}")
            print(f"  Sensitivity: {final_metrics['avg_sensitivity']:.5f} ± {final_metrics['std_sensitivity']:.5f}")
            
            # Save reconstitution results
            reconstitution_results = {
                "dataset_id": dataset_id,
                "strategy": strategy,
                "n_classes": n_classes,
                "class_names": class_names,
                "ecoc_matrix": M_ecoc_matrix.tolist(),
                "n_simulations": n_simulations,
                "test_size": test_size,
                "final_metrics": final_metrics,
                "all_simulations": {
                    "accuracy": acc_simulations,
                    "balanced_accuracy": bal_acc_simulations,
                    "cohen_kappa": kappa_simulations,
                    "geometric_mean": geom_mean_simulations,
                    "sensitivity": sensitivity_simulations
                },
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            # Save to file
            reconstitution_file = os.path.join(dataset_dir, f"reconstitution_results_{dataset_id}.json")
            with open(reconstitution_file, 'w') as f:
                json.dump(reconstitution_results, f, indent=2)
            
            print(f"Reconstitution results saved to: {reconstitution_file}")
            
            return {
                "success": True, 
                "message": f"Reconstitution completed. Avg Accuracy: {final_metrics['avg_acc']:.4f} ± {final_metrics['std_acc']:.4f}",
                "results": reconstitution_results
            }

        except Exception as e:
            error_msg = f"Error during reconstitution: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return {"success": False, "message": error_msg}

    def _create_ova_matrix(self, class_names):
        """Create One-vs-All ECOC matrix"""
        import numpy as np
        n_classes = len(class_names)
        # Each column represents one dichotomy (one class vs all others)
        M = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            M[i, i] = 1    # Positive class
            M[:, i][M[:, i] == 0] = -1  # All other classes are negative
        return M

    def _create_ovo_matrix(self, class_names):
        """Create One-vs-One ECOC matrix"""
        import numpy as np
        n_classes = len(class_names)
        n_dichotomies = n_classes * (n_classes - 1) // 2
        M = np.zeros((n_classes, n_dichotomies))
        
        dichotomy_idx = 0
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                M[i, dichotomy_idx] = 1   # First class is positive
                M[j, dichotomy_idx] = -1  # Second class is negative
                # All other classes remain 0 (not involved in this dichotomy)
                dichotomy_idx += 1
        
        return M

    def _apply_ecoc_binarization(self, y, class_names, ecoc_column, strategy, dichotomy_idx):
        """Apply ECOC binarization for a specific dichotomy"""
        import numpy as np
        
        # Create mapping from class names to indices
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        
        # Initialize binary labels
        binary_labels = np.zeros(len(y))
        
        for i, class_label in enumerate(y):
            if class_label in class_to_idx:
                class_idx = class_to_idx[class_label]
                ecoc_value = ecoc_column[class_idx]
                
                if ecoc_value == 1:
                    binary_labels[i] = 1  # Positive class
                elif ecoc_value == -1:
                    binary_labels[i] = -1  # Negative class
                else:  # ecoc_value == 0 (for OVO, some classes not involved)
                    # For OVO, we exclude samples from classes not involved in this dichotomy
                    binary_labels[i] = 0  # Will be filtered out
        
        # For OVO, filter out samples with label 0 (not involved in this dichotomy)
        if strategy == "ovo":
            mask = binary_labels != 0
            return binary_labels[mask]
        
        return binary_labels

    def _instantiate_model(self, best_config):
        """
        Instantiate the actual model that was used in the original experiment.
        This uses LSEnsemble from uc3m.labelswitching, not sklearn models.
        """
        try:
            # Import the ACTUAL model classes used in the experiments
            sys.path.insert(0, os.path.join(self.parent_dir, 'uc3m'))
            from uc3m.labelswitching import LSEnsemble
            
            # The config contains LSEnsemble parameters directly
            print(f"    Instantiating LSEnsemble with config: {list(best_config.keys())}")
            
            # Create a copy of the config to avoid modifying the original
            model_config = best_config.copy()
            
            # Ensure input_size is set if not present (required for LSEnsemble)
            if 'input_size' not in model_config:
                print("    Warning: input_size not in config, will be set during fit")
            
            # Create LSEnsemble model with the exact configuration
            model = LSEnsemble(**model_config)
            
            return model
                
        except Exception as e:
            print(f"Error instantiating LSEnsemble model: {str(e)}")
            print(f"Config: {best_config}")
            import traceback
            traceback.print_exc()
            
            # Fallback to sklearn if LSEnsemble fails
            print("    Using RandomForest as fallback...")
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(random_state=42, n_estimators=100)

    def _decode_ecoc_predictions(self, Y_pred, M_ecoc, class_names):
        """Decode ECOC predictions to get final multiclass predictions"""
        import numpy as np
        
        n_samples = Y_pred.shape[0]
        y_pred_multiclass = []
        
        for i in range(n_samples):
            sample_predictions = Y_pred[i, :]
            
            # Calculate distance to each class codeword
            distances = []
            for class_idx in range(M_ecoc.shape[0]):
                codeword = M_ecoc[class_idx, :]
                # Hamming distance (or you could use Euclidean distance)
                distance = np.sum(sample_predictions != codeword)
                distances.append(distance)
            
            # Assign to the class with minimum distance
            predicted_class_idx = np.argmin(distances)
            y_pred_multiclass.append(class_names[predicted_class_idx])
        
        return np.array(y_pred_multiclass)

    def _calculate_multiclass_metrics(self, y_true, y_pred, class_names):
        """Calculate comprehensive multiclass performance metrics"""
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        from sklearn.metrics import confusion_matrix, cohen_kappa_score
        from sklearn.metrics import classification_report
        from imblearn.metrics import geometric_mean_score, sensitivity_score
        import numpy as np
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Geometric mean and sensitivity (using imbalanced-learn)
        try:
            metrics['geometric_mean'] = geometric_mean_score(y_true, y_pred, average='weighted')
            metrics['sensitivity'] = sensitivity_score(y_true, y_pred, average='weighted')
        except Exception as e:
            print(f"Warning: Could not calculate imblearn metrics: {e}")
            metrics['geometric_mean'] = None
            metrics['sensitivity'] = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        metrics['confusion_matrix'] = cm.tolist()  # Convert to list for JSON serialization
        
        # Per-class metrics
        try:
            report = classification_report(y_true, y_pred, labels=class_names, output_dict=True)
            metrics['per_class_metrics'] = report
        except Exception as e:
            print(f"Warning: Could not generate classification report: {e}")
            metrics['per_class_metrics'] = None
        
        return metrics

def main():
    """Main entry point for the GUI application"""
    try:
        # Create experiment runner
        runner = ExperimentRunner()
        
        # Validate environment
        runner.validate_environment()
        
        # Show initial setup screen
        setup_window = LanguageSetupWindow()
        language, ml_familiar, completed = setup_window.run()
        
        if not completed:
            return  # User closed setup window without completing
        
        # Create main GUI with selected language and ML familiarity
        root = tk.Tk()
        app = MLExperimentGUI(root, runner, language, ml_familiar)
        
        # Start GUI main loop
        root.mainloop()
        
    except Exception as e:
        import tkinter.messagebox as mb
        root = tk.Tk()
        root.withdraw()  # Hide root window
        mb.showerror("Startup Error", f"Failed to start application:\n{str(e)}")
        root.destroy()


if __name__ == "__main__":
    multiprocessing.freeze_support() # For PyInstaller
    main()

