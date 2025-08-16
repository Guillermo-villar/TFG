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

# Add parent directory to import run_test
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import run_test
from gui_interface import MLExperimentGUI


class ExperimentRunner:
    """Handles the connection between GUI and run_test.py"""
    
    def __init__(self):
        self.config_path = os.path.join(parent_dir, "config.yaml")
        self.experiment_process = None
    
    def run_experiment(self, level, use_previous):
        """Execute the experiment with given parameters"""
        try:
            # Create and start a new process for the experiment
            self.experiment_process = multiprocessing.Process(target=run_test.main, args=(level, use_previous))
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

