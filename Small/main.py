#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for the ML Experiment GUI application.
"""

import tkinter as tk
import multiprocessing
import sys
import os

# Add project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from connector.experiment_runner import ExperimentRunner
from gui.setup_window import LanguageSetupWindow
from gui.main_window import MLExperimentGUI

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
        # Create a temporary root to show the error message
        temp_root = tk.Tk()
        temp_root.withdraw()  # Hide root window
        mb.showerror("Startup Error", f"Failed to start application:\n{str(e)}")
        temp_root.destroy()


if __name__ == "__main__":
    # This is important for PyInstaller to work correctly
    multiprocessing.freeze_support()
    main()
