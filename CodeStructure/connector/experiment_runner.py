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

from model import run_test
from model import run_mc
from gui.main_window import MLExperimentGUI
from gui.setup_window import LanguageSetupWindow
from connector.dataset_manager import get_dataset_status, delete_dataset_progress
from connector.report_manager import generate_experiment_report, diagnose_path_mismatch, reconstitute_multiclass_models


class ExperimentRunner:
    """Handles the connection between GUI and run_test.py"""
    
    def __init__(self):
        self.parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(self.parent_dir, "model", "config.yaml")
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
            from model import data_generator
            
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
                    # Always store the original file path, not any processed version
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
        return delete_dataset_progress(self.config_path, data_source, csv_path)

    def get_dataset_status(self, data_source="synthetic", csv_path=None, current_level=1):
        return get_dataset_status(self.config_path, data_source, csv_path, current_level)

    def generate_experiment_report(self, data_source="synthetic", csv_path=None, current_level=1, gui_root=None, language: str = "en"):
        return generate_experiment_report(self.parent_dir, self.config_path, data_source, csv_path, current_level, gui_root, language, self)

    def diagnose_path_mismatch(self, original_csv_path):
        return diagnose_path_mismatch(original_csv_path)

    def get_dataset_id(self, csv_path):
        """Get the dataset ID for a given CSV file."""
        from model.data_generator import generate_real_data_id
        return generate_real_data_id(csv_path)

    def get_dataset_paths(self, dataset_id):
        """Get the directory structure for a given dataset ID."""
        from model.data_generator import get_dataset_directory_structure
        return get_dataset_directory_structure(dataset_id)

    def reconstitute_multiclass_models(self, dataset_id):
        return reconstitute_multiclass_models(dataset_id)

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

