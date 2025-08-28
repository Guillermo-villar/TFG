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
            
            # Delete best_runner JSON file if it exists
            if os.path.exists(best_runner_file):
                os.remove(best_runner_file)
                deleted_items.append("Best runner configuration")
            
            # Delete runs directory if it exists
            if os.path.exists(runs_dir):
                shutil.rmtree(runs_dir)
                deleted_items.append("All run history")
            
            # If the entire dataset directory is empty now (except for the dataset CSV), we can optionally remove it
            # But we'll keep the dataset CSV file itself as it might be needed
            
            if deleted_items:
                return {
                    "success": True,
                    "message": f"Successfully deleted: {', '.join(deleted_items)}",
                    "dataset_id": dataset_id,
                    "deleted_items": deleted_items
                }
            else:
                return {
                    "success": True,
                    "message": "No progress data found to delete (dataset was already fresh)",
                    "dataset_id": dataset_id,
                    "deleted_items": []
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
            
            # Get dataset directory structure
            dataset_paths = data_generator.get_dataset_directory_structure(dataset_id)
            best_runner_file = dataset_paths["best_runner_json"]
            
            # Check if best_runner file exists
            if not os.path.exists(best_runner_file):
                return {
                    "status": "new",
                    "dataset_id": dataset_id,
                    "message": "Dataset has not been studied before",
                    "current_capilaridad_config": current_capilaridad_config,
                    "current_level": current_level
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
                    "current_level": current_level
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
                "current_level": current_level
            }

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

