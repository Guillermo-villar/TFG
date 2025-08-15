#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI Interface for ML Experiments
Manages all Tkinter widgets and user interface
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox


class MLExperimentGUI:
    """Main GUI interface for ML experiments"""
    
    def __init__(self, root, experiment_runner):
        self.root = root
        self.experiment_runner = experiment_runner
        self.root.title("ML Experiment Launcher")
        self.root.geometry("600x500")
        
        # State variables
        self.is_running = False
        
        # GUI variables
        self.level_var = tk.IntVar(value=1)
        self.use_previous_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready")
        
        # Create widgets
        self.create_widgets()
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        """Create all GUI widgets"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="ML Experiment Launcher", 
                               font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # Level selection
        self.create_level_selection(main_frame)
        
        # Options
        self.create_options_section(main_frame)
        
        # Control buttons and status
        self.create_controls(main_frame)
        
        # Log display
        self.create_log_display(main_frame)
    
    def create_level_selection(self, parent):
        """Create level selection widgets"""
        level_frame = ttk.LabelFrame(parent, text="Analysis Complexity Level", padding="10")
        level_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        level_frame.columnconfigure(0, weight=1)
        
        levels = [
            (1, "Level 1: Basic", "Fast execution with few combinations"),
            (2, "Level 2: Intermediate", "Balanced approach with moderate combinations"),
            (3, "Level 3: Advanced", "Exhaustive search with many combinations")
        ]
        
        for i, (level, title, description) in enumerate(levels):
            frame = ttk.Frame(level_frame)
            frame.grid(row=i, column=0, sticky=(tk.W, tk.E), pady=2)
            
            ttk.Radiobutton(frame, text=title, variable=self.level_var, 
                           value=level).pack(anchor='w')
            ttk.Label(frame, text=description, foreground='gray', 
                     font=('Arial', 8)).pack(anchor='w', padx=(20, 0))
    
    def create_options_section(self, parent):
        """Create options section"""
        options_frame = ttk.LabelFrame(parent, text="Options", padding="10")
        options_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Checkbutton(options_frame, text="Use previous best runner configuration",
                       variable=self.use_previous_var).pack(anchor='w')
    
    def create_controls(self, parent):
        """Create control buttons and status"""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        control_frame.columnconfigure(1, weight=1)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=0, column=0, sticky=tk.W)
        
        self.start_button = ttk.Button(button_frame, text="Start Experiment", 
                                      command=self.start_experiment)
        self.start_button.pack(side='left', padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="Stop", 
                                     command=self.stop_experiment, state='disabled')
        self.stop_button.pack(side='left', padx=(0, 10))
        
        ttk.Button(button_frame, text="View Config", 
                  command=self.view_config).pack(side='left')
        
        # Status
        status_frame = ttk.Frame(control_frame)
        status_frame.grid(row=0, column=1, sticky=tk.E)
        
        ttk.Label(status_frame, text="Status:").pack(side='left', padx=(0, 5))
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                font=('Arial', 10, 'bold'))
        status_label.pack(side='left')
    
    def create_log_display(self, parent):
        """Create log display area"""
        log_frame = ttk.LabelFrame(parent, text="Execution Log", padding="5")
        log_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Log controls
        log_controls = ttk.Frame(log_frame)
        log_controls.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Button(log_controls, text="Clear Log", 
                  command=self.clear_log).pack(side='left', padx=(0, 10))
    
    def start_experiment(self):
        """Start the ML experiment"""
        if self.is_running:
            messagebox.showwarning("Warning", "An experiment is already running!")
            return
        
        # Get parameters
        level = self.level_var.get()
        use_previous = self.use_previous_var.get()
        
        # Update UI
        self.is_running = True
        self.start_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.status_var.set("Running")
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        
        # Log start message
        self.add_log_message(f"Starting experiment with Level {level}, Use Previous: {use_previous}")
        
        # Schedule experiment to run in main thread after UI update
        self.root.after(100, self.run_experiment_main_thread, level, use_previous)
    
    def run_experiment_main_thread(self, level, use_previous):
        """Run experiment in main thread to avoid signal issues"""
        try:
            # Update GUI to show processing
            self.root.update()
            
            # Use the experiment runner to execute
            result = self.experiment_runner.run_experiment(level, use_previous)
            
            # Update UI on completion
            self.experiment_completed(result)
            
        except Exception as e:
            error_msg = f"Error running experiment: {str(e)}"
            self.experiment_error(error_msg)
    
    def experiment_completed(self, result):
        """Handle experiment completion"""
        self.is_running = False
        self.start_button.configure(state='normal')
        self.stop_button.configure(state='disabled')
        self.status_var.set("Completed")
        
        self.add_log_message("Experiment completed successfully!")
        messagebox.showinfo("Success", "Experiment completed successfully!")
    
    def experiment_error(self, error_msg):
        """Handle experiment error"""
        self.is_running = False
        self.start_button.configure(state='normal')
        self.stop_button.configure(state='disabled')
        self.status_var.set("Error")
        
        self.add_log_message(f"ERROR: {error_msg}")
        messagebox.showerror("Error", error_msg)
    
    def stop_experiment(self):
        """Stop the current experiment"""
        if self.is_running:
            self.add_log_message("Stop requested by user...")
            # Note: Actual stopping would require modification to run_test.py
            # For now, just update UI
            self.is_running = False
            self.start_button.configure(state='normal')
            self.stop_button.configure(state='disabled')
            self.status_var.set("Stopped")
    
    def add_log_message(self, message):
        """Add message to log display"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
    
    def clear_log(self):
        """Clear the log display"""
        self.log_text.delete(1.0, tk.END)
    
    def view_config(self):
        """Show current configuration"""
        try:
            result = self.experiment_runner.get_config_preview()
            if result:
                # Create a simple dialog to show config
                config_window = tk.Toplevel(self.root)
                config_window.title("Configuration Preview")
                config_window.geometry("500x400")
                
                text_widget = scrolledtext.ScrolledText(config_window, wrap=tk.WORD)
                text_widget.pack(fill='both', expand=True, padx=10, pady=10)
                
                text_widget.insert(1.0, result)
                text_widget.configure(state='disabled')
            else:
                messagebox.showwarning("Warning", "Configuration file not found!")
        except Exception as e:
            messagebox.showerror("Error", f"Error reading configuration: {e}")
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            if messagebox.askokcancel("Quit", "An experiment is running. Do you want to quit anyway?"):
                self.root.destroy()
        else:
            self.root.destroy()







