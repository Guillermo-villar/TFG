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
        self.study_mode_var = tk.StringVar(value="full_study")
        self.popup_log_text = None  # For the popup window
        
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
        main_frame.rowconfigure(5, weight=1)
        
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
        
        # Advanced options
        self.create_advanced_options(main_frame)
    
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
        control_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
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
        
        # Logs button
        self.view_logs_button = ttk.Button(button_frame, text="View Logs", command=self.show_logs_window)
        self.view_logs_button.pack(side="left", padx=5)

    def create_log_display(self, parent):
        """Create log display area"""
        log_frame = ttk.LabelFrame(parent, text="Execution Log", padding="5")
        log_frame.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.main_log_text = scrolledtext.ScrolledText(log_frame, height=12, wrap=tk.WORD)
        self.main_log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Log controls
        log_controls = ttk.Frame(log_frame)
        log_controls.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Button(log_controls, text="Clear Log", 
                  command=self.clear_log).pack(side='left', padx=(0, 10))
    
    def create_advanced_options(self, parent):
        """Create advanced options section"""
        advanced_frame = ttk.LabelFrame(parent, text="Advanced Settings", padding="10")
        advanced_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Radiobutton(advanced_frame, text="Study Base ML Model (Iterate Stage 1 & 2)",
                       variable=self.study_mode_var, value="full_study").pack(anchor='w')
        ttk.Radiobutton(advanced_frame, text="Only Study Class Imbalance Parameters (Iterate Stage 2)",
                       variable=self.study_mode_var, value="stage2_only").pack(anchor='w')
    
    def show_logs_window(self):
        """Creates and shows a window for displaying logs."""
        if hasattr(self, 'log_window') and self.log_window.winfo_exists():
            self.log_window.lift()
            return

        self.log_window = tk.Toplevel(self.root)
        self.log_window.title("Experiment Logs")
        self.log_window.geometry("800x600")

        log_frame = ttk.Frame(self.log_window)
        log_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.popup_log_text = tk.Text(log_frame, wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(log_frame, command=self.popup_log_text.yview)
        self.popup_log_text.config(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.popup_log_text.pack(side="left", fill="both", expand=True)

        button_frame = ttk.Frame(self.log_window)
        button_frame.pack(pady=5)

        refresh_button = ttk.Button(button_frame, text="Refresh", command=self.update_logs)
        refresh_button.pack(side="left", padx=5)
        
        close_button = ttk.Button(button_frame, text="Close", command=self.log_window.destroy)
        close_button.pack(side="left", padx=5)

        self.update_logs()

    def update_logs(self):
        """Fetches the latest logs and updates the text widget."""
        if not hasattr(self, 'log_window') or not self.log_window.winfo_exists():
            return
            
        log_content = self.experiment_runner.get_latest_log_content()
        self.popup_log_text.config(state="normal")
        self.popup_log_text.delete("1.0", tk.END)
        self.popup_log_text.insert(tk.END, log_content)
        self.popup_log_text.config(state="disabled")
        self.popup_log_text.see(tk.END) # Auto-scroll to the bottom

    def start_experiment(self):
        """Start the ML experiment"""
        if self.is_running:
            messagebox.showwarning("Warning", "An experiment is already running!")
            return
        
        level = self.level_var.get()
        use_previous = self.use_previous_var.get()
        study_mode = self.study_mode_var.get()
        
        self.is_running = True
        self.start_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.status_var.set("Running")
        
        self.main_log_text.delete(1.0, tk.END)
        self.add_log_message(f"Starting experiment with Level {level}, Use Previous: {use_previous}, Study Mode: {study_mode}")
        
        try:
            result = self.experiment_runner.run_experiment(level, use_previous, study_mode)
            self.add_log_message(result)
            self.check_experiment_status() # Start polling
        except Exception as e:
            self.experiment_error(str(e))

    def check_experiment_status(self):
        """Periodically check the status of the experiment process."""
        if not self.is_running:
            return

        process = self.experiment_runner.experiment_process
        if process and process.is_alive():
            # Process is still running, check again later
            self.root.after(1000, self.check_experiment_status)
        else:
            # Process has finished
            if process and process.exitcode == 0:
                self.experiment_completed("Experiment finished successfully.")
            elif process:
                self.experiment_error(f"Experiment process terminated with exit code {process.exitcode}.")
            # If process is None (was stopped), do nothing as UI is already updated.

    def experiment_completed(self, message):
        """Handle experiment completion"""
        self.is_running = False
        self.start_button.configure(state='normal')
        self.stop_button.configure(state='disabled')
        self.status_var.set("Completed")
        
        self.add_log_message(message)
        messagebox.showinfo("Success", message)
    
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
            self.add_log_message("Stopping experiment...")
            try:
                result = self.experiment_runner.stop_experiment()
                self.add_log_message(result)
            except Exception as e:
                self.add_log_message(f"Error stopping experiment: {str(e)}")
            
            self.is_running = False
            self.start_button.configure(state='normal')
            self.stop_button.configure(state='disabled')
            self.status_var.set("Stopped by user")
        else:
            messagebox.showinfo("Info", "No experiment is currently running.")
    
    def add_log_message(self, message):
        """Add message to log display"""
        self.main_log_text.insert(tk.END, f"{message}\n")
        self.main_log_text.see(tk.END)
    
    def clear_log(self):
        """Clear the log display"""
        self.main_log_text.delete(1.0, tk.END)
    
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
            if messagebox.askokcancel("Quit", "An experiment is running. Do you want to stop it and quit?"):
                self.stop_experiment()
                self.root.destroy()
        else:
            self.root.destroy()







