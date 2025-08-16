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
from gui_interface import MLExperimentGUI


class ExperimentRunner:
    """Handles the connection between GUI and run_test.py"""
    
    def __init__(self):
        self.parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(self.parent_dir, "config.yaml")
        self.experiment_process = None
        self.latest_log_file = None
    
    def run_experiment(self, level, use_previous, study_mode="full_study", data_source="synthetic", csv_path=None):
        """Execute the experiment with given parameters"""
        try:
            # Modify config based on data source
            self.update_config(data_source, csv_path)

            # Reset latest log file on new run
            self.latest_log_file = None
            # Create and start a new process for the experiment
            self.experiment_process = multiprocessing.Process(target=run_test.main, args=(level, use_previous, study_mode))
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
        """Finds the latest log file and returns its content."""
        if self.latest_log_file and os.path.exists(self.latest_log_file):
            with open(self.latest_log_file, 'r') as f:
                return f.read()

        results_dir = os.path.join(self.parent_dir, 'datasets', 'results')
        if not os.path.isdir(results_dir):
            return "Results directory not found."

        try:
            # Find the most recent experiment directory
            all_run_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
            if not all_run_dirs:
                return "No experiment logs found."
            
            latest_dir = max(all_run_dirs, key=lambda d: os.path.getmtime(os.path.join(results_dir, d)))
            log_file = os.path.join(results_dir, latest_dir, 'terminal_output.txt')

            if os.path.exists(log_file):
                self.latest_log_file = log_file
                with open(log_file, 'r') as f:
                    return f.read()
            else:
                return f"Log file not found in the latest run directory: {latest_dir}"
        except Exception as e:
            return f"Error reading log file: {str(e)}"

    def find_csv_files(self):
        """Find all .csv files in the project directory"""
        csv_files = []
        for root, _, files in os.walk(self.parent_dir):
            for file in files:
                if file.endswith(".csv"):
                    csv_files.append(os.path.join(root, file))
        return csv_files

    def update_config(self, data_source, csv_path):
        """Update config.yaml based on data source selection"""
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


def main():
    """Main entry point for the GUI application"""
    try:
        # Create experiment runner
        runner = ExperimentRunner()
        
        # Validate environment
        runner.validate_environment()
        
        # Create GUI
        root = tk.Tk()
        app = MLExperimentGUI(root, runner)
        
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

