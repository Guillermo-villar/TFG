#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI Interface for ML Experiments
Manages all Tkinter widgets and user interface
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import json
import os

class LanguageSetupWindow:
    """Initial window for language and ML familiarity selection"""
    
    def __init__(self):
        self.language = "en"
        self.ml_familiar = False
        self.completed = False
        
        self.root = tk.Tk()
        
        # Variables
        self.language_var = tk.StringVar(value="en")
        self.ml_familiar_var = tk.BooleanVar(value=False)
        
        # Load texts
        self.texts = self.get_texts()
        
        # Set window title with translation
        self.root.title(self.texts.get("setup_title", "ML Experiment Launcher - Setup"))
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        self.continue_button = None  # Store reference to continue button
        
        self.create_widgets()
        
        # Center window after widgets are created
        self.center_window()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def get_texts(self):
        """Get text translations based on current language"""
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            translations_file = os.path.join(script_dir, 'translations.json')
            
            with open(translations_file, 'r', encoding='utf-8') as f:
                translations = json.load(f)
            
            return translations.get(self.language, translations.get('en', {}))
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            # Fallback to minimal English if file not found or corrupted
            return {
                "welcome_title": "Welcome to ML Experiment Launcher",
                "language_selection": "Select Language / Seleccionar Idioma",
                "ml_familiarity": "Machine Learning Familiarity",
                "ml_question": "Are you familiar with Machine Learning terms and concepts?",
                "yes_familiar": "Yes, I'm familiar",
                "no_beginner": "No, I'm a beginner",
                "continue": "Continue",
                "english_option": "EN (English)",
                "spanish_option": "ES (Español)"
            }
    
    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        
        # Get the window's width and height after widgets are created
        width = self.root.winfo_reqwidth()
        height = self.root.winfo_reqheight()
        
        # Ensure minimum size
        width = max(width, 600)
        height = max(height, 500)
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate position to center the window
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        
        # Set the geometry
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
        # Make sure the window is visible
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
    
    def create_widgets(self):
        """Create setup widgets"""
        main_frame = ttk.Frame(self.root, padding="30")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        self.welcome_title_label = ttk.Label(main_frame, text=self.texts.get("welcome_title", "Welcome to ML Experiment Launcher"), 
                         font=('Arial', 16, 'bold'))
        self.welcome_title_label.pack(pady=(0, 30))
        
        # Language selection
        lang_frame = ttk.LabelFrame(main_frame, text=self.texts.get("language_selection", "Select Language / Seleccionar Idioma"), 
                                   padding="20")
        lang_frame.pack(fill='x', pady=(0, 20))
        
        buttons_frame = ttk.Frame(lang_frame)
        buttons_frame.pack()
        
        # English
        en_frame = ttk.Frame(buttons_frame)
        en_frame.pack(side='left', padx=20)
        ttk.Radiobutton(en_frame, text=self.texts.get("english_option", "EN (English)"), variable=self.language_var, 
                       value="en", command=self.update_ml_question).pack()
        
        # Spanish  
        es_frame = ttk.Frame(buttons_frame)
        es_frame.pack(side='left', padx=20)
        ttk.Radiobutton(es_frame, text=self.texts.get("spanish_option", "ES (Español)"), variable=self.language_var, 
                       value="es", command=self.update_ml_question).pack()
        
        # ML familiarity
        self.ml_frame = ttk.LabelFrame(main_frame, text="", padding="20")
        self.ml_frame.pack(fill='x', pady=(0, 30))
        
        self.ml_question_label = ttk.Label(self.ml_frame, text="", 
                                          font=('Arial', 11), wraplength=400)
        self.ml_question_label.pack(pady=(0, 15))
        
        self.buttons_frame = ttk.Frame(self.ml_frame)
        self.buttons_frame.pack()
        
        self.yes_button = ttk.Radiobutton(self.buttons_frame, text="", 
                                         variable=self.ml_familiar_var, value=True)
        self.yes_button.pack(side='left', padx=10)
        
        self.no_button = ttk.Radiobutton(self.buttons_frame, text="", 
                                        variable=self.ml_familiar_var, value=False)
        self.no_button.pack(side='left', padx=10)
        
        # Continue button
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(20, 0))
        
        self.continue_button = tk.Button(
            button_frame,
            text=self.texts.get("continue", "Continue"),
            command=self.continue_setup,
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=30,
            pady=12,
            cursor="hand2"
        )
        self.continue_button.pack(pady=10)
        
        # Update ML question to populate initial text
        self.update_ml_question()
    
    def update_ml_question(self):
        """Update ML question based on language"""
        # Reload texts for the selected language
        self.language = self.language_var.get()
        self.texts = self.get_texts()
        
        # Update window title
        self.root.title(self.texts.get("setup_title", "ML Experiment Launcher - Setup"))
        
        # Update welcome title
        self.welcome_title_label.configure(text=self.texts.get("welcome_title", "Welcome to ML Experiment Launcher"))
        
        if self.language == "es":
            self.ml_frame.configure(text=self.texts.get("ml_familiarity", "Familiaridad con Machine Learning"))
            self.ml_question_label.configure(
                text=self.texts.get("ml_question", "¿Estás familiarizado con los términos y conceptos de Machine Learning?")
            )
            self.yes_button.configure(text=self.texts.get("yes_familiar", "Sí, estoy familiarizado"))
            self.no_button.configure(text=self.texts.get("no_beginner", "No, soy principiante"))
        else:
            self.ml_frame.configure(text=self.texts.get("ml_familiarity", "Machine Learning Familiarity"))
            self.ml_question_label.configure(
                text=self.texts.get("ml_question", "Are you familiar with Machine Learning terms and concepts?")
            )
            self.yes_button.configure(text=self.texts.get("yes_familiar", "Yes, I'm familiar"))
            self.no_button.configure(text=self.texts.get("no_beginner", "No, I'm a beginner"))
        
        # Update continue button text when language changes
        if self.continue_button:
            self.continue_button.config(text=self.texts.get("continue", "Continue"))
            
        # Force the window to update its display
        self.root.update_idletasks()
    
    def continue_setup(self):
        """Continue to main app"""
        self.language = self.language_var.get()
        self.ml_familiar = self.ml_familiar_var.get()
        print(f"Setup completed with language: {self.language}, ML familiar: {self.ml_familiar}")
        self.completed = True
        self.root.destroy()
    
    def on_closing(self):
        """Handle window close"""
        self.root.destroy()
    
    def run(self):
        """Run the setup window"""
        self.root.mainloop()
        return self.language, self.ml_familiar, self.completed


class MLExperimentGUI:
    """Main GUI interface for ML experiments"""
    
    def __init__(self, root, experiment_runner, language="en", ml_familiar=False):
        self.root = root
        self.experiment_runner = experiment_runner
        self.language = language
        self.ml_familiar = ml_familiar
        
        # Load texts based on language
        self.texts = self.get_texts()
        
        self.root.title(self.texts["title"])
        self.root.geometry("800x620")
        
        # State variables
        self.is_running = False
        
        # GUI variables
        self.level_var = tk.IntVar(value=1)
        self.use_previous_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value=self.texts["ready"])
        self.study_mode_var = tk.StringVar(value="full_study")
        self.data_source_var = tk.StringVar(value="synthetic")
        self.csv_path_var = tk.StringVar(value="")
        self.popup_log_text = None  # For the popup window
        
        # Store references to widgets that need translation updates
        self.title_label = None
        
        # Create widgets
        self.create_widgets()
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def get_texts(self):
        """Get text translations based on current language"""
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            translations_file = os.path.join(script_dir, 'translations.json')
            
            with open(translations_file, 'r', encoding='utf-8') as f:
                translations = json.load(f)
            
            # Debug: Print what language we're loading and what title we got
            selected_texts = translations.get(self.language, translations.get('en', {}))
            print(f"Loading language: {self.language}")
            print(f"Title loaded: {selected_texts.get('title', 'TITLE NOT FOUND')}")
            
            return selected_texts
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Error loading translations: {e}")
            # Fallback to minimal English if file not found or corrupted
            return {
                "title": "ML Experiment Launcher",
                "error": "Error",
                "warning": "Warning",
                "ready": "Ready",
                "running": "Running",
                "completed": "Completed",
                "start_experiment": "Start Experiment",
                "stop": "Stop"
            }
    
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
        self.title_label = ttk.Label(main_frame, text=self.texts["title"], 
                               font=('Arial', 14, 'bold'))
        self.title_label.grid(row=0, column=0, pady=(0, 20), sticky='w')
        
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
        level_frame = ttk.LabelFrame(parent, text=self.texts["level_section"], padding="10")
        level_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        level_frame.columnconfigure(0, weight=1)
        
        levels = [
            (1, self.texts["level_1"], self.texts["level_1_desc"]),
            (2, self.texts["level_2"], self.texts["level_2_desc"]),
            (3, self.texts["level_3"], self.texts["level_3_desc"])
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
        data_frame = ttk.LabelFrame(parent, text=self.texts["data_source"], padding="10")
        data_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        data_frame.columnconfigure(1, weight=1)

        ttk.Radiobutton(data_frame, text=self.texts["synthetic_data"], variable=self.data_source_var, 
                        value="synthetic", command=self.toggle_csv_options).pack(anchor='w')
        
        real_data_frame = ttk.Frame(data_frame)
        real_data_frame.pack(anchor='w', fill='x', pady=(5, 0))

        ttk.Radiobutton(real_data_frame, text=self.texts["real_csv_data"], variable=self.data_source_var, 
                        value="real", command=self.toggle_csv_options).pack(side='left', anchor='w')

        self.csv_options_frame = ttk.Frame(real_data_frame)

        # Frame for file selection widgets
        file_selection_frame = ttk.Frame(self.csv_options_frame)
        file_selection_frame.pack(anchor='w', fill='x', pady=(5, 0))

        # This frame will contain the button and the label/dropdown part
        left_frame = ttk.Frame(file_selection_frame)
        left_frame.pack(side='left', anchor='n')
        
        browse_button = ttk.Button(left_frame, text=self.texts["select_file"], command=self.browse_csv_file)
        browse_button.pack(side='left', padx=(0, 5))

        right_frame = ttk.Frame(file_selection_frame)
        right_frame.pack(side='left', expand=True, fill='x')

        # New label for standardized tests
        ttk.Label(right_frame, text=self.texts["standard_dataset"]).pack(anchor='w')

        self.csv_path_dropdown = ttk.Combobox(right_frame, textvariable=self.csv_path_var, width=30)
        self.csv_path_dropdown.pack(anchor='w', expand=True, fill='x')
        self.csv_path_dropdown.set(self.texts["see_standard_tests"])

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
            messagebox.showerror(self.texts["error"], f"Failed to find CSV files: {e}")

    def browse_csv_file(self):
        """Open file dialog to select a CSV file"""
        filepath = filedialog.askopenfilename(
            title=self.texts["select_csv_title"],
            filetypes=((self.texts["csv_files"], "*.csv"), (self.texts["all_files"], "*.*"))
        )
        if filepath:
            current_values = list(self.csv_path_dropdown['values'])
            if filepath not in current_values:
                current_values.append(filepath)
                self.csv_path_dropdown['values'] = current_values
            self.csv_path_var.set(filepath)
    
    def create_options_section(self, parent):
        """Create options section"""
        options_frame = ttk.LabelFrame(parent, text=self.texts["options"], padding="10")
        options_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Checkbutton(options_frame, text=self.texts["use_previous"],
                       variable=self.use_previous_var).pack(anchor='w')

    def create_advanced_options(self, parent):
        """Create advanced options section"""
        advanced_frame = ttk.LabelFrame(parent, text=self.texts["advanced_settings"], padding="10")
        advanced_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Radiobutton(advanced_frame, text=self.texts["study_base_ml"],
                       variable=self.study_mode_var, value="full_study").pack(anchor='w')
        ttk.Radiobutton(advanced_frame, text=self.texts["study_imbalance"],
                       variable=self.study_mode_var, value="stage2_only").pack(anchor='w')

    def create_controls(self, parent):
        """Create control buttons and status"""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(10, 10))
        control_frame.columnconfigure(1, weight=1)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=0, column=0, sticky=tk.W)
        
        self.start_button = ttk.Button(button_frame, text=self.texts["start_experiment"], 
                                      command=self.start_experiment)
        self.start_button.pack(side='left', padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text=self.texts["stop"], 
                                     command=self.stop_experiment, state='disabled')
        self.stop_button.pack(side='left', padx=(0, 10))
        
        ttk.Button(button_frame, text=self.texts["view_config"], 
                  command=self.view_config).pack(side='left')
        
        # Status
        status_frame = ttk.Frame(control_frame)
        status_frame.grid(row=0, column=1, sticky=tk.E)
        
        ttk.Label(status_frame, text=self.texts["status"]).pack(side='left', padx=(0, 5))
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                font=('Arial', 10, 'bold'))
        status_label.pack(side='left')
        
        # Logs button
        self.view_logs_button = ttk.Button(button_frame, text=self.texts["view_logs"], command=self.show_logs_window)
        self.view_logs_button.pack(side="left", padx=5)

    def create_log_display(self, parent):
        """Create log display area"""
        log_frame = ttk.LabelFrame(parent, text=self.texts["execution_log"], padding="5")
        log_frame.grid(row=6, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.main_log_text = scrolledtext.ScrolledText(log_frame, height=12, wrap=tk.WORD)
        self.main_log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Log controls
        log_controls = ttk.Frame(log_frame)
        log_controls.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Button(log_controls, text=self.texts["clear_log"], 
                  command=self.clear_log).pack(side='left', padx=(0, 10))
    
    def show_logs_window(self):
        """Creates and shows a window for displaying logs."""
        if hasattr(self, 'log_window') and self.log_window.winfo_exists():
            self.log_window.lift()
            return

        self.log_window = tk.Toplevel(self.root)
        self.log_window.title(self.texts["experiment_logs"])
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

        refresh_button = ttk.Button(button_frame, text=self.texts["refresh"], command=self.update_logs)
        refresh_button.pack(side="left", padx=5)
        
        close_button = ttk.Button(button_frame, text=self.texts["close"], command=self.log_window.destroy)
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
            messagebox.showwarning(self.texts["warning"], self.texts["experiment_running"])
            return
        
        level = self.level_var.get()
        use_previous = self.use_previous_var.get()
        study_mode = self.study_mode_var.get()
        data_source = self.data_source_var.get()
        csv_path = self.csv_path_var.get() if data_source == "real" else None
        
        self.is_running = True
        self.start_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.status_var.set(self.texts["running"])
        
        self.main_log_text.delete(1.0, tk.END)
        self.add_log_message(self.texts["starting_experiment"].format(level, use_previous, study_mode))
        if data_source == 'real':
            self.add_log_message(self.texts["using_real_data"].format(csv_path))
        else:
            self.add_log_message(self.texts["using_synthetic_data"])

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
                self.experiment_completed(self.texts["experiment_finished"])
            elif process:
                self.experiment_error(self.texts["experiment_terminated"].format(process.exitcode))
            # If process is None (was stopped), do nothing as UI is already updated.

    def experiment_completed(self, message):
        """Handle experiment completion"""
        self.is_running = False
        self.start_button.configure(state='normal')
        self.stop_button.configure(state='disabled')
        self.status_var.set(self.texts["completed"])
        
        self.add_log_message(message)
        messagebox.showinfo(self.texts["success"], message)
    
    def experiment_error(self, error_msg):
        """Handle experiment error"""
        self.is_running = False
        self.start_button.configure(state='normal')
        self.stop_button.configure(state='disabled')
        self.status_var.set(self.texts["error"])
        
        self.add_log_message(f"ERROR: {error_msg}")
        messagebox.showerror(self.texts["error"], error_msg)
    
    def stop_experiment(self):
        """Stop the current experiment"""
        if self.is_running:
            self.add_log_message(self.texts["stopping_experiment"])
            try:
                result = self.experiment_runner.stop_experiment()
                self.add_log_message(result)
            except Exception as e:
                self.add_log_message(f"Error stopping experiment: {str(e)}")
            
            self.is_running = False
            self.start_button.configure(state='normal')
            self.stop_button.configure(state='disabled')
            self.status_var.set(self.texts["stopped"])
        else:
            messagebox.showinfo(self.texts["info"], self.texts["no_experiment"])
    
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
                config_window.title(self.texts["config_preview"])
                config_window.geometry("500x400")
                
                text_widget = scrolledtext.ScrolledText(config_window, wrap=tk.WORD)
                text_widget.pack(fill='both', expand=True, padx=10, pady=10)
                
                text_widget.insert(1.0, result)
                text_widget.configure(state='disabled')
            else:
                messagebox.showwarning(self.texts["warning"], self.texts["config_not_found"])
        except Exception as e:
            messagebox.showerror(self.texts["error"], f"Error reading configuration: {e}")
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            if messagebox.askokcancel("Quit", self.texts["quit_confirm"]):
                self.stop_experiment()
                self.root.destroy()
        else:
            self.root.destroy()
            messagebox.showinfo(self.texts["info"], self.texts["no_experiment"])
    
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
                config_window.title(self.texts["config_preview"])
                config_window.geometry("500x400")
                
                text_widget = scrolledtext.ScrolledText(config_window, wrap=tk.WORD)
                text_widget.pack(fill='both', expand=True, padx=10, pady=10)
                
                text_widget.insert(1.0, result)
                text_widget.configure(state='disabled')
            else:
                messagebox.showwarning(self.texts["warning"], self.texts["config_not_found"])
        except Exception as e:
            messagebox.showerror(self.texts["error"], f"Error reading configuration: {e}")
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            if messagebox.askokcancel("Quit", self.texts["quit_confirm"]):
                self.stop_experiment()
                self.root.destroy()
        else:
            self.root.destroy()







