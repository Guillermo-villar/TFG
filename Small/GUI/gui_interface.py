#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI Interface for ML Experiments
Manages all Tkinter widgets and user interface
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog


class MLExperimentGUI:
    """Main GUI interface for ML experiments"""
    
    def __init__(self, root, experiment_runner):
        self.root = root
        self.experiment_runner = experiment_runner
        self.root.title("ML Experiment Launcher")
        self.root.geometry("800x620")
        
        # State variables
        self.is_running = False
        
        # GUI variables
        self.level_var = tk.IntVar(value=1)
        self.use_previous_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready")
        self.study_mode_var = tk.StringVar(value="full_study")
        self.data_source_var = tk.StringVar(value="synthetic")
        self.csv_path_var = tk.StringVar(value="")
        self.popup_log_text = None  # For the popup window
        
        # Create widgets
        self.create_widgets()
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Create a canvas and a scrollbar
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        main_frame = ttk.Frame(scrollable_frame, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(6, weight=1) # Adjusted row for log display
        
        # Title
        title_label = ttk.Label(main_frame, text="ML Experiment Launcher", 
                               font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 20), sticky='w')
        
        # Level selection
        self.create_level_selection(main_frame)
        
        # Data source selection
        self.create_data_source_selection(main_frame)

        # Options
        self.create_options_section(main_frame)

        # Advanced options
        self.create_advanced_options(main_frame)
        
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
    
    def create_data_source_selection(self, parent):
        """Create data source selection widgets"""
        data_frame = ttk.LabelFrame(parent, text="Data Source", padding="10")
        data_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        data_frame.columnconfigure(1, weight=1)

        ttk.Radiobutton(data_frame, text="Synthetic Data", variable=self.data_source_var, 
                        value="synthetic", command=self.toggle_csv_options).pack(anchor='w')
        
        real_data_frame = ttk.Frame(data_frame)
        real_data_frame.pack(anchor='w', fill='x', pady=(5, 0))

        ttk.Radiobutton(real_data_frame, text="Real CSV Data", variable=self.data_source_var, 
                        value="real", command=self.toggle_csv_options).pack(side='left', anchor='w')

        self.csv_options_frame = ttk.Frame(real_data_frame)

        # Frame for file selection widgets
        file_selection_frame = ttk.Frame(self.csv_options_frame)
        file_selection_frame.pack(anchor='w', fill='x', pady=(5, 0))

        # This frame will contain the button and the label/dropdown part
        left_frame = ttk.Frame(file_selection_frame)
        left_frame.pack(side='left', anchor='n')
        
        browse_button = ttk.Button(left_frame, text="📁 Select a file", command=self.browse_csv_file)
        browse_button.pack(side='left', padx=(0, 5))

        right_frame = ttk.Frame(file_selection_frame)
        right_frame.pack(side='left', expand=True, fill='x')

        # New label for standardized tests
        ttk.Label(right_frame, text="Or test with an industry-standard dataset").pack(anchor='w')

        self.csv_path_dropdown = ttk.Combobox(right_frame, textvariable=self.csv_path_var, width=30)
        self.csv_path_dropdown.pack(anchor='w', expand=True, fill='x')
        self.csv_path_dropdown.set("See standardized tests...")

        self.populate_csv_files()
        self.csv_options_frame.pack(side='left', padx=(10, 0), expand=True, fill='x')

    def toggle_csv_options(self):
        """Enable/disable CSV options based on radio button selection"""
        is_real_data = self.data_source_var.get() == "real"
        
        # Iterate over child widgets of csv_options_frame to enable/disable them
        for child in self.csv_options_frame.winfo_children():
            try:
                # This will work for ttk widgets
                child.configure(state='normal' if is_real_data else 'disabled')
            except tk.TclError:
                # Some widgets like Frames might not have a 'state' option
                pass
        
        # Specifically handle Combobox, which needs a different way to disable
        self.csv_path_dropdown.configure(state='readonly' if is_real_data else 'disabled')
        
        # Also handle widgets inside nested frames
        for child in self.csv_options_frame.winfo_children():
            if isinstance(child, ttk.Frame):
                for grandchild in child.winfo_children():
                    try:
                        grandchild.configure(state='normal' if is_real_data else 'disabled')
                    except tk.TclError:
                        pass

    def populate_csv_files(self):
        """Populate the dropdown with available CSV files"""
        try:
            csv_files = self.experiment_runner.find_csv_files()
            # Add MNIST placeholders
            mnist_files = ["mnist_train.csv", "mnist_test.csv"]
            for f in mnist_files:
                if f not in csv_files:
                    csv_files.insert(0, f)
            
            self.csv_path_dropdown['values'] = csv_files
            if csv_files:
                self.csv_path_var.set(csv_files[0])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to find CSV files: {e}")

    def browse_csv_file(self):
        """Open file dialog to select a CSV file"""
        filepath = filedialog.askopenfilename(
            title="Select your CSV file",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if filepath:
            current_values = list(self.csv_path_dropdown['values'])
            if filepath not in current_values:
                current_values.append(filepath)
                self.csv_path_dropdown['values'] = current_values
            self.csv_path_var.set(filepath)
    
    def create_options_section(self, parent):
        """Create options section"""
        options_frame = ttk.LabelFrame(parent, text="Options", padding="10")
        options_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Checkbutton(options_frame, text="Use previous best runner configuration",
                       variable=self.use_previous_var).pack(anchor='w')

    def create_advanced_options(self, parent):
        """Create advanced options section"""
        advanced_frame = ttk.LabelFrame(parent, text="Advanced Settings", padding="10")
        advanced_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Radiobutton(advanced_frame, text="Study Base ML Model (Iterate Stage 1 & 2)",
                       variable=self.study_mode_var, value="full_study").pack(anchor='w')
        ttk.Radiobutton(advanced_frame, text="Only Study Class Imbalance Parameters (Iterate Stage 2)",
                       variable=self.study_mode_var, value="stage2_only").pack(anchor='w')

    def create_controls(self, parent):
        """Create control buttons and status"""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(10, 10))
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
        log_frame.grid(row=6, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.main_log_text = scrolledtext.ScrolledText(log_frame, height=12, wrap=tk.WORD)
        self.main_log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Log controls
        log_controls = ttk.Frame(log_frame)
        log_controls.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Button(log_controls, text="Clear Log", 
                  command=self.clear_log).pack(side='left', padx=(0, 10))
    
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
        data_source = self.data_source_var.get()
        csv_path = self.csv_path_var.get() if data_source == "real" else None
        
        self.is_running = True
        self.start_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.status_var.set("Running")
        
        self.main_log_text.delete(1.0, tk.END)
        self.add_log_message(f"Starting experiment with Level {level}, Use Previous: {use_previous}, Study Mode: {study_mode}")
        if data_source == 'real':
            self.add_log_message(f"Using real data from: {csv_path}")
        else:
            self.add_log_message("Using synthetic data")

        try:
            result = self.experiment_runner.run_experiment(
                level=level, 
                use_previous=use_previous, 
                study_mode=study_mode,
                data_source=data_source,
                csv_path=csv_path
            )
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







