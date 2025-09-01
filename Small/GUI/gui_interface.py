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
import datetime

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
                "spanish_option": "ES (Español)",
                "advanced_access_title": "Advanced Configuration Access",
                "advanced_access_message": "Great! As an experienced ML user, you will have access to advanced configurations in the application. You can run experiments with full control over every configuration aspect available in config.yaml."
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
                                         variable=self.ml_familiar_var, value=True,
                                         command=self.on_ml_familiar_selected)
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
    
    def on_ml_familiar_selected(self):
        """Handle when ML familiar option is selected"""
        if self.ml_familiar_var.get():  # Only show popup if "Yes, I'm familiar" is selected
            self.show_advanced_access_popup()
    
    def show_advanced_access_popup(self):
        """Show popup informing about advanced configuration access"""
        title = self.texts.get("advanced_access_title", "Advanced Configuration Access")
        message = self.texts.get("advanced_access_message", 
                                "Great! As an experienced ML user, you will have access to advanced configurations in the application.")
        
        messagebox.showinfo(title, message)
    
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
        self.is_starting = False  # Track initial start phase to prevent progress updates
        self.current_progress_info = None
        
        # GUI variables
        self.level_var = tk.IntVar(value=1)
        self.status_var = tk.StringVar(value=self.texts["ready"])
        self.progress_detail_var = tk.StringVar(value="")
        self.study_mode_var = tk.StringVar(value="full_study")
        self.data_source_var = tk.StringVar(value="synthetic")
        self.csv_path_var = tk.StringVar(value="")
        self.csv_full_path_display_var = tk.StringVar(value="")
        self.popup_log_text = None  # For the popup window
        
        # Separate variables for industry datasets (just for display) and user files (for actual use)
        self.industry_csv_path = ""  # Only for industry dataset selection
        self.user_csv_path = ""      # Only for user-selected files
        self.csv_file_map = {}
        
        # Multiclass detection state
        self.multiclass_strategy = None  # Will store 'ova', 'ovo', or None
        self.multiclass_detected = False
        self.multiclass_target_column = None
        self.multiclass_indicator_var = None  # Will be initialized when GUI is created
        self.multiclass_dichotomies = None  # Will store dichotomy information
        
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

        # Advanced options
        self.create_advanced_options(main_frame)
        
        # Control buttons and status
        self.create_controls(main_frame)
        
        # Log display
        self.create_log_display(main_frame)
    
    def create_level_selection(self, parent):
        """Create level selection widgets"""
        # Choose title and labels based on ML familiarity
        if self.ml_familiar:
            section_title = self.texts["level_section"]
            levels = [
                (1, self.texts["level_1"], self.texts["level_1_desc"]),
                (2, self.texts["level_2"], self.texts["level_2_desc"]),
                (3, self.texts["level_3"], self.texts["level_3_desc"])
            ]
        else:
            section_title = self.texts["simple_level_section"]
            levels = [
                (1, self.texts["simple_level_1"], self.texts["simple_level_1_desc"]),
                (2, self.texts["simple_level_2"], self.texts["simple_level_2_desc"]),
                (3, self.texts["simple_level_3"], self.texts["simple_level_3_desc"])
            ]
        
        level_frame = ttk.LabelFrame(parent, text=section_title, padding="10")
        level_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        level_frame.columnconfigure(0, weight=1)
        
        if self.ml_familiar:
            # Traditional vertical layout for ML familiar users
            for i, (level, title, description) in enumerate(levels):
                frame = ttk.Frame(level_frame)
                frame.grid(row=i, column=0, sticky=(tk.W, tk.E), pady=2)
                
                ttk.Radiobutton(frame, text=title, variable=self.level_var, 
                               value=level).pack(anchor='w')
                ttk.Label(frame, text=description, foreground='gray', 
                         font=('Arial', 8)).pack(anchor='w', padx=(20, 0))
        else:
            # Intuitive horizontal slider-like layout for beginners
            self.create_simple_level_selector(level_frame, levels)
    
    def create_simple_level_selector(self, parent, levels):
        """Create a simple, intuitive level selector for non-ML users"""
        # Instructions
        instruction_label = ttk.Label(parent, 
                                     text=self.texts["level_instruction"],
                                     font=('Arial', 10, 'bold'))
        instruction_label.grid(row=0, column=0, pady=(0, 15), sticky='w')
        
        # Create horizontal selector container
        selector_container = ttk.Frame(parent)
        selector_container.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        selector_container.columnconfigure(1, weight=1)
        
        # Create the three option frames in a horizontal line
        options_frame = ttk.Frame(selector_container)
        options_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        # Configure equal column weights for balanced spacing
        for i in range(3):
            options_frame.columnconfigure(i, weight=1)
        
        # Create option buttons with better styling
        self.level_buttons = []
        button_colors = ["#FF5722", "#FF9800", "#4CAF50"]  # Red to Green gradient
        button_texts = ["🚀", "⚖️", "🔬"]  # Icons for each level
        
        for i, (level, title, description) in enumerate(levels):
            # Create frame for each option
            option_frame = ttk.Frame(options_frame)
            option_frame.grid(row=0, column=i, padx=10, pady=10, sticky='ew')
            
            # Create styled button for selection
            level_button = tk.Button(option_frame,
                                   text=f"{button_texts[i]}\n{title}",
                                   command=lambda l=level: self.select_level(l),
                                   bg=button_colors[i],
                                   fg="white",
                                   font=("Arial", 10, "bold"),
                                   relief="raised",
                                   padx=15,
                                   pady=10,
                                   cursor="hand2",
                                   width=12,
                                   height=3,
                                   wraplength=100)
            level_button.pack(pady=(0, 5))
            
            # Store button reference for highlighting
            self.level_buttons.append(level_button)
            
            # Description label
            desc_label = ttk.Label(option_frame, 
                                  text=description,
                                  font=('Arial', 9),
                                  foreground='gray',
                                  justify='center',
                                  wraplength=120)
            desc_label.pack()
            
            # Create radio button (hidden) for form functionality
            radio = ttk.Radiobutton(option_frame, 
                                   variable=self.level_var, 
                                   value=level,
                                   command=lambda: self.update_level_buttons())
            # Hide the radio button but keep it functional
            radio.pack_forget()
        
        # Set default selection and update button states
        self.level_var.set(2)  # Default to balanced
        self.update_level_buttons()
    
    def select_level(self, level):
        """Handle level selection from simple selector"""
        self.level_var.set(level)
        self.update_level_buttons()
    
    def update_level_buttons(self):
        """Update button appearance based on selection"""
        if hasattr(self, 'level_buttons'):
            selected_level = self.level_var.get()
            button_colors = ["#FF5722", "#FF9800", "#4CAF50"]
            button_colors_selected = ["#D32F2F", "#F57C00", "#388E3C"]  # Darker versions
            
            for i, button in enumerate(self.level_buttons):
                level = i + 1
                if level == selected_level:
                    # Highlight selected button
                    button.config(bg=button_colors_selected[i], 
                                 relief="sunken",
                                 borderwidth=3)
                else:
                    # Normal state
                    button.config(bg=button_colors[i], 
                                 relief="raised",
                                 borderwidth=2)
    
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

        self.csv_path_dropdown = ttk.Combobox(right_frame, textvariable=self.csv_path_var, width=30, state="readonly")
        self.csv_path_dropdown.pack(anchor='w', expand=True, fill='x')
        
        # Label to display the full path of the selected file
        self.full_path_label = ttk.Label(right_frame, textvariable=self.csv_full_path_display_var, wraplength=400, foreground="gray")
        self.full_path_label.pack(anchor='w', expand=True, fill='x', pady=(5,0))
        
        # Add callback for when user selects a file from dropdown
        self.csv_path_dropdown.bind('<<ComboboxSelected>>', self.on_csv_file_selected)

        self.populate_csv_files()
        self.csv_options_frame.pack(side='left', padx=(10, 0), expand=True, fill='x')

    def on_csv_file_selected(self, event=None):
        """Called when user selects a CSV file from the dropdown (industry datasets only for demo)"""
        selected_display_name = self.csv_path_var.get()
        
        # Clear user manual selection when selecting industry dataset
        self.user_csv_path = ""
        
        # Get the full path from the map (this is just for industry datasets display)
        self.industry_csv_path = self.csv_file_map.get(selected_display_name, "")
        
        # For industry datasets: show only filename in dropdown, full path in display label
        if self.industry_csv_path and os.path.exists(self.industry_csv_path):
            # Automatically switch to real data mode when industry dataset is selected
            self.data_source_var.set("real")
            self.toggle_csv_options()  # Update UI state
            
            # Show just the filename in the log message  
            self.add_log_message(f"Selected industry test dataset: {selected_display_name} (for demo/testing only)")
            self.add_log_message(f"Switched to real data mode for industry dataset")
            # Show full path in the display label below dropdown
            self.csv_full_path_display_var.set(self.industry_csv_path)
        else:
            # This shouldn't happen with industry datasets, but handle gracefully
            self.add_log_message(f"Warning: Selected file does not exist: {self.industry_csv_path}")
            self.csv_full_path_display_var.set("")

    def toggle_csv_options(self):
        """Enable/disable CSV options based on radio button selection"""
        is_real_data = self.data_source_var.get() == "real"
        
        # Clear previous selections when switching modes
        if not is_real_data:
            # Switching to synthetic data - clear all CSV selections
            self.industry_csv_path = ""
            self.user_csv_path = ""
            self.csv_full_path_display_var.set("")
        
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
        """Populate the dropdown with industry test datasets only"""
        try:
            # Only show industry test datasets in dropdown
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            industry_dir = os.path.join(parent_dir, "industry")
            
            self.csv_file_map = {}
            
            if os.path.exists(industry_dir):
                industry_files = []
                for file in os.listdir(industry_dir):
                    if file.endswith('.csv'):
                        full_path = os.path.join(industry_dir, file)
                        display_name = os.path.basename(file)
                        self.csv_file_map[display_name] = full_path
                        industry_files.append(display_name)
                
                # Add industry files (sorted for consistent order)
                display_names = sorted(industry_files)
                self.csv_path_dropdown['values'] = display_names
                if display_names:
                    # Don't auto-select any dataset - let user choose
                    self.csv_path_var.set(self.texts.get("see_standard_tests", "Select a dataset to test..."))
                else:
                    self.csv_path_var.set(self.texts.get("see_standard_tests", "No datasets available"))
            else:
                self.csv_path_var.set(self.texts.get("see_standard_tests", "No datasets available"))
        except Exception as e:
            messagebox.showerror(self.texts["error"], f"Failed to find CSV files: {e}")

    def browse_csv_file(self):
        """Open file dialog to select a CSV file"""
        filepath = filedialog.askopenfilename(
            title=self.texts["select_csv_title"],
            filetypes=((self.texts["csv_files"], "*.csv"), (self.texts["all_files"], "*.*"))
        )
        if filepath:
            # Import required modules first
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from csv_preloading import CSVPreprocessor
            
            # Clear industry dataset selection when selecting manual file
            self.industry_csv_path = ""
            
            # Automatically switch to real data mode when external file is selected
            self.data_source_var.set("real")
            self.toggle_csv_options()  # Update UI state
            self.add_log_message(f"Switched to real data mode for external file: {os.path.basename(filepath)}")
            
            # Initialize CSV preprocessor for automatic preprocessing
            
            try:
                # Initialize preprocessor with GUI enabled
                preprocessor = CSVPreprocessor(gui_enabled=True)
                
                # Check if preprocessing is needed
                analysis = preprocessor.analyze_csv(filepath)
                
                # NEW: Check for multiclass candidates after analysis
                multiclass_candidates = preprocessor.detect_multiclass(filepath)
                
                if multiclass_candidates:
                    # Show multiclass detection and strategy selection
                    self.add_log_message(f"Multiclass dataset detected with {len(multiclass_candidates)} potential target column(s)")
                    
                    # For now, use the first candidate (you could enhance this to let user choose)
                    main_candidate = multiclass_candidates[0]
                    
                    # Show confirmation and strategy selection popup
                    selected_strategy = self.show_multiclass_confirmation_popup(main_candidate)
                    
                    if selected_strategy:
                        # User confirmed multiclass mode and selected a strategy
                        self.multiclass_strategy = selected_strategy
                        self.multiclass_detected = True
                        self.multiclass_target_column = main_candidate['column']
                        
                        self.add_log_message(f"Multiclass mode enabled with {selected_strategy.upper()} strategy")
                        self.add_log_message(f"Target column: '{main_candidate['column']}' with {main_candidate['n_classes']} classes")
                        
                        # Update the visual indicator
                        self.update_multiclass_indicator()
                    else:
                        # User declined multiclass mode - treat as regular preprocessing
                        self.multiclass_strategy = None
                        self.multiclass_detected = False
                        self.multiclass_target_column = None
                        self.add_log_message("Multiclass mode declined - proceeding with standard binary classification")
                        
                        # Update the visual indicator
                        self.update_multiclass_indicator()
                else:
                    # No multiclass candidates found - standard binary processing
                    self.multiclass_strategy = None
                    self.multiclass_detected = False
                    self.multiclass_target_column = None
                    
                    # Update the visual indicator
                    self.update_multiclass_indicator()
                
                # Handle processing based on multiclass detection
                if multiclass_candidates:
                    # MULTICLASS FLOW: Show multiclass-specific dialog
                    main_candidate = multiclass_candidates[0]
                    
                    # Calculate number of dichotomies based on strategy
                    n_classes = main_candidate['n_classes']
                    strategy = self.multiclass_strategy
                    if strategy == 'ovo':
                        num_dichotomies = n_classes * (n_classes - 1) // 2
                    elif strategy == 'ova':
                        num_dichotomies = n_classes
                    else:
                        num_dichotomies = 'a number of' # Fallback text

                    # Show multiclass preprocessing dialog
                    proceed = messagebox.askyesno(
                        self.texts["multiclass_dataset_detected_title"], 
                        self.texts["multiclass_dataset_detected_message"].format(
                            filename=os.path.basename(filepath),
                            n_classes=main_candidate['n_classes'],
                            class_names=', '.join(map(str, main_candidate.get('class_names', []))),
                            column=main_candidate['column'],
                            strategy=self.multiclass_strategy.upper() if self.multiclass_strategy else self.texts.get('not_selected', 'Not selected'),
                            num_dichotomies=num_dichotomies,
                            strategy_name=self.multiclass_strategy.upper() if self.multiclass_strategy else self.texts.get('selected', 'selected')
                        )
                    )
                    
                    if proceed:
                        # MULTICLASS PROCESSING: Create dichotomies
                        self.add_log_message(f"Creating multiclass dichotomies using {self.multiclass_strategy.upper()} strategy...")
                        
                        # Import and use multiclass decomposition
                        from csv_preloading import create_multiclass_dichotomies
                        
                        dichotomy_result = create_multiclass_dichotomies(
                            filepath, 
                            main_candidate, 
                            strategy=self.multiclass_strategy
                        )
                        
                        if dichotomy_result['success']:
                            n_dichotomies = dichotomy_result['n_dichotomies']
                            self.add_log_message(f"Successfully created {n_dichotomies} dichotomies")
                            
                            # Show summary of dichotomies
                            summary_text = self.texts["multiclass_decomposition_complete_message"].format(
                                strategy=dichotomy_result['strategy'],
                                n_classes=dichotomy_result['n_classes'],
                                class_names=', '.join(map(str, dichotomy_result['class_names'])),
                                n_dichotomies=n_dichotomies,
                                dichotomy_details="\n".join([
                                    f"Dichotomy {dich['index']}: {dich['n_instances']} instances\n"
                                    f"  Positive: {', '.join(dich['positive_classes'])}\n"
                                    f"  Negative: {', '.join(dich['negative_classes'])}"
                                    for i, dich in enumerate(dichotomy_result['dichotomies'][:3])
                                ]) + (f"\n... and {n_dichotomies - 3} more dichotomies" if n_dichotomies > 3 else "")
                            )
                            
                            messagebox.showinfo(self.texts["multiclass_decomposition_complete_title"], summary_text)
                            
                            # Store dichotomy information for later use
                            self.multiclass_dichotomies = dichotomy_result
                            
                            # NUEVO: Procesar cada dicotomía individualmente
                            self.add_log_message(f"Processing {n_dichotomies} dichotomies...")
                            processed_dichotomies = []
                            
                            for i, dichotomy_file in enumerate(dichotomy_result['dichotomy_files']):
                                dichotomy_name = os.path.basename(dichotomy_file)
                                self.add_log_message(f"Processing dichotomy {i+1}/{n_dichotomies}: {dichotomy_name}")
                                
                                # Create processed filename for this dichotomy
                                base_name = os.path.splitext(dichotomy_file)[0]
                                processed_dichotomy_path = f"{base_name}_processed.csv"
                                
                                try:
                                    # Process this individual dichotomy
                                    dichotomy_result_proc = preprocessor.process_csv_file(dichotomy_file, processed_dichotomy_path)
                                    
                                    if dichotomy_result_proc['success']:
                                        processed_dichotomies.append({
                                            'original': dichotomy_file,
                                            'processed': dichotomy_result_proc['output_path'],
                                            'index': i + 1,
                                            'success': True
                                        })
                                        self.add_log_message(f"  ✓ Dichotomy {i+1} processed successfully")
                                    else:
                                        error_msg = dichotomy_result_proc.get('error', 'Unknown error')
                                        processed_dichotomies.append({
                                            'original': dichotomy_file,
                                            'processed': dichotomy_file,  # Use original if processing failed
                                            'index': i + 1,
                                            'success': False,
                                            'error': error_msg
                                        })
                                        self.add_log_message(f"  ⚠ Dichotomy {i+1} processing failed: {error_msg}")
                                        
                                except Exception as e:
                                    processed_dichotomies.append({
                                        'original': dichotomy_file,
                                        'processed': dichotomy_file,  # Use original if exception
                                        'index': i + 1,
                                        'success': False,
                                        'error': str(e)
                                    })
                                    self.add_log_message(f"  ✗ Dichotomy {i+1} processing exception: {str(e)}")
                            
                            # Store processed dichotomies info
                            self.multiclass_dichotomies['processed_dichotomies'] = processed_dichotomies
                            
                            # Set the first processed dichotomy as the primary file for display
                            if processed_dichotomies:
                                first_processed = processed_dichotomies[0]['processed']
                                self.user_csv_path = first_processed
                                self.csv_path_var.set(f"Multiclass ({n_dichotomies} dichotomies)")
                                self.csv_full_path_display_var.set(first_processed)
                                self.add_log_message(f"Ready to run experiments on {n_dichotomies} processed dichotomies")
                                
                                # Show processing summary
                                successful_count = sum(1 for d in processed_dichotomies if d['success'])
                                messagebox.showinfo(self.texts["dichotomy_processing_complete_title"], 
                                                  self.texts["dichotomy_processing_complete_message"].format(
                                                      successful=successful_count, 
                                                      total=n_dichotomies
                                                  ))
                            else:
                                self.add_log_message("No dichotomies could be processed")
                                messagebox.showerror(self.texts["processing_failed_title"], 
                                                   self.texts["processing_failed_message"])
                        else:
                            # Dichotomy creation failed
                            error_msg = dichotomy_result.get('error', 'Unknown error in multiclass decomposition')
                            self.add_log_message(f"Error creating dichotomies: {error_msg}")
                            messagebox.showerror(self.texts["multiclass_decomposition_failed_title"], 
                                               self.texts["multiclass_decomposition_failed_message"].format(error=error_msg))
                            return
                    else:
                        # User declined multiclass processing
                        self.add_log_message("Multiclass processing declined - using original file")
                        self.user_csv_path = filepath
                        self.csv_path_var.set(filepath)
                        self.csv_full_path_display_var.set(filepath)
                
                else:
                    # NOT MULTICLASS: Standard binary processing flow
                    # Create processed filename
                    base_name = os.path.splitext(os.path.basename(filepath))[0]
                    processed_path = os.path.join(os.path.dirname(filepath), f"{base_name}_processed.csv")
                    
                    # Show standard preprocessing dialog
                    proceed = messagebox.askyesno(
                        self.texts["csv_preprocessing_title"], 
                        self.texts["csv_preprocessing_message"].format(
                            original_file=os.path.basename(filepath),
                            processed_file=os.path.basename(processed_path),
                            num_columns=len(analysis['columns']),
                            binary_columns=len(analysis['binary_columns']),
                            text_columns=len(analysis.get('text_columns', [])),
                            categorical_columns=len(analysis.get('categorical_columns', []))
                        )
                    )
                    
                    if proceed:
                        # STANDARD BINARY PROCESSING
                        result = preprocessor.process_csv_file(filepath, processed_path)
                        
                        if result['success']:
                            final_path = result['output_path']
                            self.add_log_message(f"CSV preprocessing completed: {os.path.basename(final_path)}")
                            
                            # Set the processed file as the user's selected file
                            self.user_csv_path = final_path
                            self.csv_path_var.set(final_path)  # Show full path for manual files
                            self.csv_full_path_display_var.set(final_path)  # Show full path in display label
                            
                            # Show preprocessing summary
                            log_summary = "\n".join(result['processing_log'])
                            if log_summary:
                                messagebox.showinfo(self.texts["preprocessing_complete_title"], 
                                                  self.texts["preprocessing_complete_message"].format(summary=log_summary))
                        else:
                            # Preprocessing failed
                            error_msg = result.get('error', 'Unknown preprocessing error')
                            messagebox.showerror(self.texts["preprocessing_failed_title"], 
                                               self.texts["preprocessing_failed_message"].format(error=error_msg))
                            # Set original file as the user's selected file
                            self.user_csv_path = filepath
                            self.csv_path_var.set(filepath)  # Show full path for manual files
                            self.csv_full_path_display_var.set(filepath)  # Show full path in display label
                    else:
                        # User declined preprocessing, set original file as the user's selected file
                        self.user_csv_path = filepath
                        self.csv_path_var.set(filepath)  # Show full path for manual files
                        self.csv_full_path_display_var.set(filepath)  # Show full path in display label
                    
            except Exception as e:
                # If preprocessing fails, fall back to original file
                error_details = str(e)
                if "text_columns" in error_details:
                    error_msg = "CSV analysis error: Missing expected data structure. This has been fixed - please try again."
                elif "encoding" in error_details:
                    error_msg = f"CSV encoding error: {error_details}. Try saving the CSV with UTF-8 encoding."
                elif "pandas" in error_details:
                    error_msg = f"CSV format error: {error_details}. Please check that the file is a valid CSV."
                else:
                    error_msg = f"Preprocessing error: {error_details}"
                    
                messagebox.showerror(self.texts["preprocessing_error_title"], 
                                   self.texts["preprocessing_error_message"].format(error=error_msg))
                # Set original file as the user's selected file
                self.user_csv_path = filepath
                self.csv_path_var.set(filepath)  # Show full path for manual files
                self.csv_full_path_display_var.set(filepath)  # Show full path in display label
    
    def show_multiclass_confirmation_popup(self, candidate):
        """
        Show popup to confirm multiclass detection and ask user for strategy selection
        
        Parameters:
        -----------
        candidate : dict
            Dictionary containing multiclass candidate information
            
        Returns:
        --------
        str or None : Selected strategy ('ova', 'ovo') or None if cancelled
        """
        # First, confirm with user that this is indeed a multiclass dataset
        confirm_message = self.texts["multiclass_confirmation_message"].format(
            column=candidate['column'],
            n_classes=candidate['n_classes'],
            class_names=', '.join(map(str, candidate['class_names'][:5])) + ('...' if len(candidate['class_names']) > 5 else '')
        )
        
        confirmed = messagebox.askyesno(self.texts["multiclass_confirmation_title"], confirm_message)
        
        if not confirmed:
            return None
            
        # If confirmed, show strategy selection popup
        return self.show_decomposition_strategy_popup()
    
    def show_decomposition_strategy_popup(self):
        """
        Show popup for decomposition strategy selection (OVA vs OVO)
        
        Returns:
        --------
        str or None : Selected strategy ('ova', 'ovo') or None if cancelled
        """
        # Create custom dialog
        dialog = tk.Toplevel(self.root)
        dialog.title(self.texts["decomposition_strategy_title"])
        dialog.geometry("500x450")
        dialog.transient(self.root)
        
        # Try to set grab, but handle gracefully if it fails
        try:
            dialog.grab_set()
        except tk.TclError:
            # If grab fails, continue without it - the dialog will still work
            pass
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        result = {'strategy': None}
        
        # Main title
        title_label = tk.Label(dialog, 
                              text=self.texts["decomposition_strategy_header"],
                              font=('Arial', 14, 'bold'))
        title_label.pack(pady=20)
        
        # Strategy selection variable
        strategy_var = tk.StringVar(value="ova")
        
        # Create frames for each strategy
        ova_frame = ttk.LabelFrame(dialog, text=self.texts["ova_strategy_title"], padding=10)
        ova_frame.pack(fill='x', padx=20, pady=10)
        
        ova_radio = ttk.Radiobutton(ova_frame, text=self.texts["ova_strategy_select"], 
                                   variable=strategy_var, value="ova")
        ova_radio.pack(anchor='w')
        
        ova_desc = tk.Label(ova_frame, 
                           text=self.texts["ova_strategy_description"],
                           justify='left', font=('Arial', 9))
        ova_desc.pack(anchor='w', pady=(5, 0))
        
        ovo_frame = ttk.LabelFrame(dialog, text=self.texts["ovo_strategy_title"], padding=10)
        ovo_frame.pack(fill='x', padx=20, pady=10)
        
        ovo_radio = ttk.Radiobutton(ovo_frame, text=self.texts["ovo_strategy_select"], 
                                   variable=strategy_var, value="ovo")
        ovo_radio.pack(anchor='w')
        
        ovo_desc = tk.Label(ovo_frame, 
                           text=self.texts["ovo_strategy_description"],
                           justify='left', font=('Arial', 9))
        ovo_desc.pack(anchor='w', pady=(5, 0))
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)
        
        def confirm():
            result['strategy'] = strategy_var.get()
            dialog.destroy()
        
        def cancel():
            result['strategy'] = None
            dialog.destroy()
        
        ttk.Button(button_frame, text=self.texts["confirm"], command=confirm).pack(side='left', padx=10)
        ttk.Button(button_frame, text=self.texts["cancel"], command=cancel).pack(side='left', padx=10)
        
        # Wait for dialog to close
        dialog.wait_window()
        
        return result['strategy']

    def update_multiclass_indicator(self):
        """Update the multiclass mode indicator in the status bar"""
        if self.multiclass_detected and self.multiclass_strategy:
            indicator_text = f"[MULTICLASS: {self.multiclass_strategy.upper()}]"
            self.multiclass_indicator_var.set(indicator_text)
            # Show the multiclass strategy button when multiclass is detected
            self.multiclass_button.pack(side='left', padx=(0, 10))
        else:
            self.multiclass_indicator_var.set("")
            # Hide the multiclass strategy button when not in multiclass mode
            self.multiclass_button.pack_forget()

    def change_multiclass_strategy(self):
        """Allow user to change the multiclass decomposition strategy"""
        if not self.multiclass_detected:
            messagebox.showwarning(self.texts["warning"], 
                                 self.texts["multiclass_strategy_change_warning"])
            return
        
        # Show the strategy selection popup
        new_strategy = self.show_decomposition_strategy_popup()
        
        if new_strategy and new_strategy != self.multiclass_strategy:
            # Update the strategy
            old_strategy = self.multiclass_strategy
            self.multiclass_strategy = new_strategy
            
            # Update the indicator
            self.update_multiclass_indicator()
            
            # Log the change
            self.add_log_message(f"Multiclass strategy changed from {old_strategy.upper()} to {new_strategy.upper()}")
            
            # Show confirmation
            messagebox.showinfo(self.texts["multiclass_strategy_changed_title"], 
                              self.texts["multiclass_strategy_changed_message"].format(strategy=new_strategy.upper()))
        elif new_strategy == self.multiclass_strategy:
            # User selected the same strategy
            self.add_log_message(f"Multiclass strategy remains {new_strategy.upper()}")
        # If new_strategy is None, user cancelled - do nothing

    def create_advanced_options(self, parent):
        """Create advanced options section"""
        advanced_frame = ttk.LabelFrame(parent, text=self.texts["advanced_settings"], padding="10")
        advanced_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Radiobutton(advanced_frame, text=self.texts["study_base_ml"],
                       variable=self.study_mode_var, value="full_study").pack(anchor='w')
        ttk.Radiobutton(advanced_frame, text=self.texts["study_imbalance"],
                       variable=self.study_mode_var, value="stage2_only").pack(anchor='w')

    def create_controls(self, parent):
        """Create control buttons and status"""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(10, 10))
        control_frame.columnconfigure(1, weight=1)
        
        # Buttons with colors and icons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=0, column=0, sticky=tk.W)
        
        # Start button - Green with play icon
        self.start_button = tk.Button(button_frame, 
                                     text=f"▶ {self.texts['start_experiment']}", 
                                     command=self.start_experiment,
                                     bg="#4CAF50",  # Green
                                     fg="white",
                                     font=("Arial", 10, "bold"),
                                     relief="raised",
                                     padx=15,
                                     pady=8,
                                     cursor="hand2")
        self.start_button.pack(side='left', padx=(0, 10))
        
        # Stop button - Red with stop icon
        self.stop_button = tk.Button(button_frame, 
                                    text=f"⏹ {self.texts['stop']}", 
                                    command=self.stop_experiment, 
                                    state='disabled',
                                    bg="#f44336",  # Red
                                    fg="white",
                                    font=("Arial", 10, "bold"),
                                    relief="raised",
                                    padx=15,
                                    pady=8,
                                    cursor="hand2",
                                    disabledforeground="#cccccc")
        self.stop_button.pack(side='left', padx=(0, 10))
        
        # View Config button - Blue
        self.config_button = tk.Button(button_frame, 
                                      text=f"⚙ {self.texts['view_config']}", 
                                      command=self.view_config,
                                      bg="#2196F3",  # Blue
                                      fg="white",
                                      font=("Arial", 10),
                                      relief="raised",
                                      padx=10,
                                      pady=8,
                                      cursor="hand2")
        self.config_button.pack(side='left', padx=(0, 10))
        
        # Multiclass Strategy button - Purple (initially hidden)
        self.multiclass_button = tk.Button(button_frame, 
                                          text="🎯 Multiclass Strategy", 
                                          command=self.change_multiclass_strategy,
                                          bg="#9C27B0",  # Purple
                                          fg="white",
                                          font=("Arial", 10),
                                          relief="raised",
                                          padx=10,
                                          pady=8,
                                          cursor="hand2")
        # Initially hidden - will be shown when multiclass is detected
        
        # Logs button - Orange with magnifying glass icon
        self.view_logs_button = tk.Button(button_frame, 
                                         text=f"🔍 {self.texts['view_logs']}", 
                                         command=self.show_logs_window,
                                         bg="#FF9800",  # Orange
                                         fg="white",
                                         font=("Arial", 10),
                                         relief="raised",
                                         padx=10,
                                         pady=8,
                                         cursor="hand2")
        self.view_logs_button.pack(side="left", padx=(0, 10))
        
        # Status and Progress Frame
        status_progress_frame = ttk.Frame(control_frame)
        status_progress_frame.grid(row=0, column=1, sticky=(tk.W, tk.E))
        status_progress_frame.columnconfigure(0, weight=1)
        
        # Main status line
        status_line_frame = ttk.Frame(status_progress_frame)
        status_line_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        status_line_frame.columnconfigure(1, weight=1)
        
        ttk.Label(status_line_frame, text=self.texts["status"]).pack(side='left', padx=(0, 5))
        status_label = ttk.Label(status_line_frame, textvariable=self.status_var, 
                                font=('Arial', 10, 'bold'))
        status_label.pack(side='left')
        
        # Multiclass mode indicator
        self.multiclass_indicator_var = tk.StringVar(value="")
        self.multiclass_indicator = ttk.Label(status_line_frame, textvariable=self.multiclass_indicator_var,
                                             font=('Arial', 9, 'bold'), foreground='blue')
        self.multiclass_indicator.pack(side='left', padx=(10, 0))
        
        # Progress detail line (ETA, runs, etc.)
        progress_detail_label = ttk.Label(status_progress_frame, textvariable=self.progress_detail_var,
                                         font=('Arial', 9), foreground='gray')
        progress_detail_label.grid(row=1, column=0, sticky=tk.W)
        
        # Custom progress bar
        self.create_custom_progress_bar(status_progress_frame)

    def create_custom_progress_bar(self, parent):
        """Create custom progress bar that shows both Stage 1 and Stage 2"""
        # Canvas for custom progress bar
        self.progress_canvas = tk.Canvas(parent, height=25, bg='white', 
                                       relief='sunken', bd=1)
        self.progress_canvas.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(2, 0))
        
        # Configure canvas to expand with window
        parent.grid_columnconfigure(0, weight=1)
        
        # Initialize progress values
        self.current_stage1_progress = "0/0"
        self.current_stage2_progress = "0/0"
        self.current_stage1_complete = False
        self.current_experiment_complete = False
        
        # Draw initial empty progress bar
        self.update_custom_progress_bar("0/0", "0/0", False, False)
    
    def update_custom_progress_bar(self, stage1_progress, stage2_progress, stage1_complete, experiment_complete=False):
        """Update the custom progress bar with stage-specific progress"""
        # Store current values
        self.current_stage1_progress = stage1_progress
        self.current_stage2_progress = stage2_progress
        self.current_stage1_complete = stage1_complete
        self.current_experiment_complete = experiment_complete
        
        # Clear canvas
        self.progress_canvas.delete("all")
        
        # Get canvas dimensions
        canvas_width = self.progress_canvas.winfo_width()
        canvas_height = self.progress_canvas.winfo_height()
        
        # Use fallback dimensions if canvas not yet rendered
        if canvas_width <= 1:
            canvas_width = 300
        if canvas_height <= 1:
            canvas_height = 23
        
        # Parse progress values
        try:
            stage1_current, stage1_total = map(int, stage1_progress.split('/'))
            stage2_current, stage2_total = map(int, stage2_progress.split('/'))
        except (ValueError, AttributeError):
            stage1_current = stage1_total = stage2_current = stage2_total = 0
        
        # Check if this is a multiclass experiment (stage2_total = 0 indicates multiclass)
        is_multiclass = (stage2_total == 0 and stage1_total > 0)
        
        if is_multiclass:
            # Multiclass mode: show single progress bar for dichotomies
            self.draw_multiclass_progress_bar(canvas_width, canvas_height, stage1_current, stage1_total, stage1_complete)
        else:
            # Binary mode: show two-stage progress bar
            self.draw_binary_progress_bar(canvas_width, canvas_height, stage1_current, stage1_total, 
                                        stage2_current, stage2_total, stage1_complete, experiment_complete)
        
        # Bind resize event to redraw progress bar
        self.progress_canvas.bind('<Configure>', self._on_canvas_resize)

    def draw_multiclass_progress_bar(self, canvas_width, canvas_height, current, total, complete):
        """Draw progress bar for multiclass experiments (dichotomies)"""
        if total == 0:
            return
        
        progress_ratio = current / total
        
        # Colors for multiclass
        if complete:
            fill_color = "#4CAF50"  # Green when complete
            outline_color = "#2E7D32"
        else:
            fill_color = "#9C27B0"  # Purple for multiclass in progress
            outline_color = "#6A1B9A"
        
        background_color = "#E0E0E0"
        
        # Draw background
        self.progress_canvas.create_rectangle(2, 2, canvas_width-2, canvas_height-2, 
                                            fill=background_color, outline="#BDBDBD", width=1)
        
        # Draw progress fill
        if progress_ratio > 0:
            fill_width = (canvas_width - 4) * progress_ratio
            self.progress_canvas.create_rectangle(2, 2, 2 + fill_width, canvas_height-2,
                                                fill=fill_color, outline=outline_color, width=2)
            
            # Add text if there's enough space
            if progress_ratio >= 0.25:
                text_x = fill_width / 2 + 2
                text = "Dichotomies" if progress_ratio >= 0.4 else "MC"
                self.progress_canvas.create_text(text_x, canvas_height/2, 
                                               text=text, fill="white", 
                                               font=("Arial", 9, "bold"))

    def draw_binary_progress_bar(self, canvas_width, canvas_height, stage1_current, stage1_total,
                               stage2_current, stage2_total, stage1_complete, experiment_complete):
        """Draw progress bar for binary experiments (two stages)"""
        # Calculate total work and progress
        total_work = stage1_total + stage2_total
        total_completed = stage1_current + stage2_current
        
        if total_work == 0:
            return  # Nothing to draw
        
        # Calculate stage boundaries
        stage1_width_ratio = stage1_total / total_work if total_work > 0 else 0
        stage1_boundary = canvas_width * stage1_width_ratio
        
        # Colors
        stage1_color = "#4CAF50" if stage1_complete else "#2196F3"  # Green if complete, blue if active
        stage1_outline = "#2E7D32" if stage1_complete else "#1976D2"  # Darker outline
        
        # Stage 2 colors: Green if experiment is completely finished, blue otherwise
        stage2_color = "#4CAF50" if experiment_complete else "#2196F3"  
        stage2_outline = "#2E7D32" if experiment_complete else "#1976D2"
        
        background_color = "#E0E0E0"  # Light gray background
        
        # Draw background
        self.progress_canvas.create_rectangle(2, 2, canvas_width-2, canvas_height-2, 
                                            fill=background_color, outline="#BDBDBD", width=1)
        
        # Draw Stage 1 progress
        if stage1_total > 0:
            stage1_progress_ratio = stage1_current / stage1_total
            stage1_fill_width = stage1_boundary * stage1_progress_ratio
            
            if stage1_fill_width > 0:
                self.progress_canvas.create_rectangle(2, 2, stage1_fill_width, canvas_height-2,
                                                    fill=stage1_color, outline=stage1_outline, width=2)
                
                # Add "Stage 1" text only if there's at least 25% progress
                if stage1_progress_ratio >= 0.25:
                    text_x = min(stage1_fill_width / 2, stage1_boundary / 2)
                    self.progress_canvas.create_text(text_x, canvas_height/2, 
                                                   text="Stage 1", fill="white", 
                                                   font=("Arial", 9, "bold"))
        
        # Draw Stage 2 progress (only if Stage 1 is complete and Stage 2 has started)
        if stage1_complete and stage2_total > 0 and stage2_current > 0:
            stage2_progress_ratio = stage2_current / stage2_total
            stage2_start_x = stage1_boundary
            stage2_available_width = canvas_width - stage1_boundary - 2
            stage2_fill_width = stage2_available_width * stage2_progress_ratio
            
            if stage2_fill_width > 0:
                self.progress_canvas.create_rectangle(stage2_start_x, 2, 
                                                    stage2_start_x + stage2_fill_width, canvas_height-2,
                                                    fill=stage2_color, outline=stage2_outline, width=2)
                
                # Add "Stage 2" text only if there's at least 25% progress
                if stage2_progress_ratio >= 0.25:
                    text_x = stage2_start_x + (stage2_fill_width / 2)
                    self.progress_canvas.create_text(text_x, canvas_height/2,
                                                   text="Stage 2", fill="white", 
                                                   font=("Arial", 9, "bold"))
        
        # Draw stage boundary line (subtle divider)
        if stage1_total > 0 and stage2_total > 0:
            self.progress_canvas.create_line(stage1_boundary, 2, stage1_boundary, canvas_height-2,
                                           fill="#757575", width=1)
    
    def _on_canvas_resize(self, event):
        """Handle canvas resize by redrawing the progress bar"""
        self.update_custom_progress_bar(self.current_stage1_progress, 
                                      self.current_stage2_progress, 
                                      self.current_stage1_complete,
                                      self.current_experiment_complete)

    def create_log_display(self, parent):
        """Create log display area"""
        log_frame = ttk.LabelFrame(parent, text=self.texts["execution_log"], padding="5")
        log_frame.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
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
        study_mode = self.study_mode_var.get()
        data_source = self.data_source_var.get()
        # Use either industry dataset or user's selected file path
        csv_path = None
        csv_source_type = None
        if data_source == "real":
            if self.industry_csv_path:
                csv_path = self.industry_csv_path
                csv_source_type = "industry"
            elif self.user_csv_path:
                csv_path = self.user_csv_path  
                csv_source_type = "manual"
        
        # Validate that user has selected a file when using real data
        if data_source == "real" and not csv_path:
            messagebox.showwarning(
                self.texts["warning"], 
                self.texts["csv_file_validation_warning"]
            )
            return
        
        # Check dataset status before starting
        try:
            current_level = self.level_var.get()
            dataset_status = self.experiment_runner.get_dataset_status(data_source, csv_path, current_level)
            
            # Show appropriate popup based on dataset status
            if not self.show_dataset_status_popup(dataset_status):
                return  # User cancelled after seeing the status
                
        except Exception as e:
            messagebox.showerror(self.texts["error"], f"Error checking dataset status: {str(e)}")
            return
        
        self.is_running = True
        self.start_button.configure(state='disabled', bg="#cccccc")  # Gray when disabled
        self.stop_button.configure(state='normal', bg="#f44336")     # Red when enabled
        self.status_var.set(self.texts["running"])
        
        # Initialize progress display
        self.update_custom_progress_bar("0/0", "0/0", False, False)
        self.progress_detail_var.set("Starting experiment...")
        
        # Add separator with timestamp instead of clearing log to stack experiments
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.add_log_message("=" * 60)
        self.add_log_message(f"NEW EXPERIMENT - {timestamp}")
        self.add_log_message("=" * 60)
        self.add_log_message(self.texts["starting_experiment"].format(level, study_mode))
        if data_source == 'real':
            if csv_source_type == "industry":
                filename = os.path.basename(csv_path)
                self.add_log_message(f"Using industry test dataset: {filename}")
                self.add_log_message(f"Full path: {csv_path}")
            elif csv_source_type == "manual":
                self.add_log_message(f"Using manually selected file: {csv_path}")
            else:
                self.add_log_message(self.texts["using_real_data"].format(csv_path))
        else:
            self.add_log_message(self.texts["using_synthetic_data"])

        try:
            # Pass multiclass info if available
            multiclass_dichotomies = self.multiclass_dichotomies if self.multiclass_detected else None
            
            result = self.experiment_runner.run_experiment(
                level=level, 
                study_mode=study_mode,
                data_source=data_source,
                csv_path=csv_path,
                multiclass_strategy=self.multiclass_strategy,
                multiclass_dichotomies=multiclass_dichotomies
            )
            self.add_log_message(result)
            self.check_experiment_status() # Start polling
        except Exception as e:
            self.experiment_error(str(e))

    def show_dataset_status_popup(self, dataset_status):
        """
        Show dataset status popup and return True if user wants to continue
        """
        status = dataset_status.get("status", "unknown")
        dataset_id = dataset_status.get("dataset_id", "unknown")
        
        if status == "new":
            # New dataset popup
            title = self.texts["new_dataset"]
            message = f"{self.texts['new_dataset_message']}\n\n"
            message += f"Dataset ID: {dataset_id}"
            
            return messagebox.askokcancel(title, message)
            
        elif status == "existing":
            # Existing dataset with progress tracking
            return self.show_existing_dataset_popup(dataset_status)
            
        elif status == "existing_old":
            # Old format dataset - create a simple dialog with delete option
            return self.show_old_format_dataset_popup(dataset_status)
            
        elif status == "error":
            # Error checking status
            messagebox.showerror(self.texts["error"], dataset_status.get("message", "Unknown error"))
            return False
            
        else:
            # Unknown status
            return messagebox.askokcancel(self.texts["warning"], f"Unknown dataset status: {status}")

    def show_old_format_dataset_popup(self, dataset_status):
        """Show popup for old format dataset with delete option"""
        dataset_id = dataset_status.get("dataset_id", "unknown")
        
        # Create custom dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title(self.texts["existing_dataset"])
        dialog.geometry("500x400")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        
        # Try to set grab, but handle gracefully if it fails
        try:
            dialog.grab_set()
        except tk.TclError:
            # If grab fails, continue without it - the dialog will still work
            pass
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (500 // 2)
        y = (dialog.winfo_screenheight() // 2) - (400 // 2)
        dialog.geometry(f"500x400+{x}+{y}")
        
        # Result variable
        result = tk.BooleanVar(value=False)
        
        # Main frame
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text=self.texts["existing_dataset"], 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 15))
        
        # Dataset info
        info_frame = ttk.LabelFrame(main_frame, text=self.texts["dataset_status"], padding="10")
        info_frame.pack(fill='x', pady=(0, 15))
        
        ttk.Label(info_frame, text=f"Dataset ID: {dataset_id}", 
                 font=('Arial', 10, 'bold')).pack(anchor='w')
        ttk.Label(info_frame, text="Format: Old (legacy)", 
                 font=('Arial', 9)).pack(anchor='w')
        
        # Description
        desc_text = ("This dataset has been studied before using an older format.\n\n"
                    "A best runner configuration is available from the previous study.\n\n"
                    "The experiment will use the previous best configuration to continue "
                    "from where it left off.")
        
        desc_label = ttk.Label(main_frame, text=desc_text, font=('Arial', 10), 
                              wraplength=450, justify='left')
        desc_label.pack(pady=15)
        
        # Best runner info if available
        if dataset_status.get("best_runner"):
            arch_frame = ttk.LabelFrame(main_frame, text=self.texts["best_architecture"], padding="10")
            arch_frame.pack(fill='x', pady=(0, 15))
            
            best_runner = dataset_status["best_runner"]
            if isinstance(best_runner, dict):
                arch_text = f"Previous best configuration found with optimal parameters."
                arch_label = ttk.Label(arch_frame, text=arch_text, font=('Arial', 9), wraplength=430)
                arch_label.pack(anchor='w')
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(15, 0))
        
        def on_continue():
            result.set(True)
            dialog.destroy()
        
        def on_cancel():
            result.set(False)
            dialog.destroy()
        
        def on_delete_progress():
            # Close current dialog first
            dialog.destroy()
            
            # Show deletion confirmation and execute if confirmed
            data_source = self.data_source_var.get()
            csv_path = self.get_current_csv_path() if data_source == "real" else None
            
            self.delete_dataset_progress(data_source, csv_path, dataset_status)
            
            # Set result to False so the experiment doesn't start automatically
            result.set(False)
        
        # Delete progress button (left side) - Red with trash icon
        delete_button = tk.Button(button_frame, 
                                 text=f"🗑 {self.texts['delete_progress']}", 
                                 command=on_delete_progress,
                                 bg="#e53e3e",  # Red
                                 fg="white",
                                 font=("Arial", 9, "bold"),
                                 relief="raised",
                                 padx=10,
                                 pady=5,
                                 cursor="hand2")
        delete_button.pack(side='left')
        
        # Continue and Cancel buttons (right side)
        continue_button = tk.Button(button_frame, 
                                   text=f"✓ {self.texts['continue']}", 
                                   command=on_continue,
                                   bg="#4CAF50",  # Green
                                   fg="white",
                                   font=("Arial", 9, "bold"),
                                   relief="raised",
                                   padx=15,
                                   pady=5,
                                   cursor="hand2")
        continue_button.pack(side='right', padx=(10, 0))
        
        cancel_button = tk.Button(button_frame, 
                                 text=f"✗ {self.texts['cancel']}", 
                                 command=on_cancel,
                                 bg="#9e9e9e",  # Gray
                                 fg="white",
                                 font=("Arial", 9),
                                 relief="raised",
                                 padx=15,
                                 pady=5,
                                 cursor="hand2")
        cancel_button.pack(side='right')
        
        # Wait for dialog to close
        dialog.wait_window()
        
        return result.get()

    def show_existing_dataset_popup(self, dataset_status):
        """Show detailed popup for existing dataset with progress tracking"""
        
        experiment_type = dataset_status.get("experiment_type", "binary")
        
        if experiment_type == "multiclass":
            return self.show_multiclass_dataset_popup(dataset_status)
        
        # Standard binary classification popup
        # Create custom dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title(self.texts["dataset_status"])
        dialog.geometry("600x650")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        
        # Try to set grab, but handle gracefully if it fails
        try:
            dialog.grab_set()
        except tk.TclError:
            # If grab fails, continue without it - the dialog will still work
            pass
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (600 // 2)
        y = (dialog.winfo_screenheight() // 2) - (650 // 2)
        dialog.geometry(f"600x650+{x}+{y}")
        
        # Result variable
        result = tk.BooleanVar(value=False)
        
        # Main frame
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text=self.texts["existing_dataset"], 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 15))
        
        # Dataset info
        info_frame = ttk.LabelFrame(main_frame, text=self.texts["dataset_status"], padding="10")
        info_frame.pack(fill='x', pady=(0, 15))
        
        ttk.Label(info_frame, text=f"Dataset ID: {dataset_status['dataset_id']}", 
                 font=('Arial', 10, 'bold')).pack(anchor='w')
        ttk.Label(info_frame, text=f"{self.texts['last_updated']}: {dataset_status['last_updated']}", 
                 font=('Arial', 9)).pack(anchor='w')
        
        # Capilaridad information
        capilaridad_frame = ttk.LabelFrame(main_frame, text=self.texts["capilaridad_level"], padding="10")
        capilaridad_frame.pack(fill='x', pady=(0, 15))
        
        # Show previous capilaridad configuration
        previous_capilaridad = dataset_status.get("previous_capilaridad_config", {})
        previous_level = dataset_status.get("previous_level", "unknown")
        
        if previous_capilaridad:
            level_names = {1: self.texts["level_basic"], 2: self.texts["level_intermediate"], 3: self.texts["level_advanced"]}
            level_name = level_names.get(int(previous_level) if str(previous_level).isdigit() else 0, str(previous_level))
            
            ttk.Label(capilaridad_frame, text=f"{self.texts['previous_capilaridad']}: {level_name} (Level {previous_level})", 
                     font=('Arial', 10, 'bold')).pack(anchor='w')
            
            max_time = previous_capilaridad.get("max_seconds_per_model", "N/A")
            n_sims = previous_capilaridad.get("n_simulations", "N/A")
            if max_time != "N/A":
                ttk.Label(capilaridad_frame, text=self.texts["max_time_per_model"].format(max_time), 
                         font=('Arial', 9)).pack(anchor='w', padx=(20, 0))
            if n_sims != "N/A":
                ttk.Label(capilaridad_frame, text=self.texts["simulations"].format(n_sims), 
                         font=('Arial', 9)).pack(anchor='w', padx=(20, 0))
        else:
            ttk.Label(capilaridad_frame, text=f"{self.texts['previous_capilaridad']}: Level {previous_level}", 
                     font=('Arial', 10, 'bold')).pack(anchor='w')
        
        # Progress info
        progress_frame = ttk.LabelFrame(main_frame, text=self.texts["progress_details"], padding="10")
        progress_frame.pack(fill='x', pady=(0, 15))
        
        # Stage 1 status
        stage1_frame = ttk.Frame(progress_frame)
        stage1_frame.pack(fill='x', pady=(0, 10))
        
        stage1_status = self.texts["study_completed"] if dataset_status["stage1_completed"] else self.texts["in_progress"]
        ttk.Label(stage1_frame, text=f"{self.texts['stage1_status']}: {stage1_status}", 
                 font=('Arial', 10, 'bold')).pack(anchor='w')
        ttk.Label(stage1_frame, text=f"{self.texts['configurations_tested']}: {dataset_status['stage1_progress']}", 
                 font=('Arial', 9)).pack(anchor='w', padx=(20, 0))
        
        # Stage 2 status
        stage2_frame = ttk.Frame(progress_frame)
        stage2_frame.pack(fill='x', pady=(0, 10))
        
        stage2_total = int(dataset_status['stage2_progress'].split('/')[1]) if '/' in dataset_status['stage2_progress'] else 0
        stage2_done = int(dataset_status['stage2_progress'].split('/')[0]) if '/' in dataset_status['stage2_progress'] else 0
        
        if stage2_total == 0:
            stage2_status = self.texts["not_started"]
        elif stage2_done >= stage2_total:
            stage2_status = self.texts["study_completed"]
        else:
            stage2_status = self.texts["in_progress"]
        
        ttk.Label(stage2_frame, text=f"{self.texts['stage2_status']}: {stage2_status}", 
                 font=('Arial', 10, 'bold')).pack(anchor='w')
        ttk.Label(stage2_frame, text=f"{self.texts['configurations_tested']}: {dataset_status['stage2_progress']}", 
                 font=('Arial', 9)).pack(anchor='w', padx=(20, 0))
        
        # Best architecture info
        if dataset_status.get("best_runner"):
            arch_frame = ttk.LabelFrame(main_frame, text=self.texts["best_architecture"], padding="10")
            arch_frame.pack(fill='x', pady=(0, 15))
            
            best_runner = dataset_status["best_runner"]
            arch_text = f"Experts: {best_runner.get('num_experts', 'N/A')}, "
            arch_text += f"Hidden Size: {best_runner.get('hidden_size', 'N/A')}, "
            arch_text += f"Dropout: {best_runner.get('drop_out', 'N/A')}, "
            arch_text += f"Batch: {best_runner.get('n_batch', 'N/A')}, "
            arch_text += f"Epochs: {best_runner.get('n_epoch', 'N/A')}"
            
            arch_label = ttk.Label(arch_frame, text=arch_text, font=('Arial', 9), wraplength=550)
            arch_label.pack(anchor='w')
        
        # Resume message
        overall_status = dataset_status.get("overall_status", "unknown")
        if overall_status == "complete":
            message = self.texts["study_complete_message"]
        else:
            message = self.texts["resume_message"]
        
        message_label = ttk.Label(main_frame, text=message, font=('Arial', 10), 
                                 wraplength=550, justify='center')
        message_label.pack(pady=15)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(15, 0))
        
        # Store dataset info for delete functionality
        dialog.dataset_status = dataset_status
        
        def on_continue():
            result.set(True)
            dialog.destroy()
        
        def on_cancel():
            result.set(False)
            dialog.destroy()
        
        def on_delete_progress():
            # Close current dialog first
            dialog.destroy()
            
            # Show deletion confirmation and execute if confirmed
            data_source = self.data_source_var.get()
            csv_path = self.get_current_csv_path() if data_source == "real" else None
            
            self.delete_dataset_progress(data_source, csv_path, dataset_status)
            
            # Set result to False so the experiment doesn't start automatically
            result.set(False)
        
        # Delete progress button (left side) - Red with trash icon
        delete_button = tk.Button(button_frame, 
                                 text=f"🗑 {self.texts['delete_progress']}", 
                                 command=on_delete_progress,
                                 bg="#e53e3e",  # Red
                                 fg="white",
                                 font=("Arial", 9, "bold"),
                                 relief="raised",
                                 padx=10,
                                 pady=5,
                                 cursor="hand2")
        delete_button.pack(side='left')
        
        # Continue and Cancel buttons (right side)
        continue_button = tk.Button(button_frame, 
                                   text=f"✓ {self.texts['continue']}", 
                                   command=on_continue,
                                   bg="#4CAF50",  # Green
                                   fg="white",
                                   font=("Arial", 9, "bold"),
                                   relief="raised",
                                   padx=15,
                                   pady=5,
                                   cursor="hand2")
        continue_button.pack(side='right', padx=(10, 0))
        
        cancel_button = tk.Button(button_frame, 
                                 text=f"✗ {self.texts['cancel']}", 
                                 command=on_cancel,
                                 bg="#9e9e9e",  # Gray
                                 fg="white",
                                 font=("Arial", 9),
                                 relief="raised",
                                 padx=15,
                                 pady=5,
                                 cursor="hand2")
        cancel_button.pack(side='right')
        
        # Wait for dialog to close
        dialog.wait_window()
        
        return result.get()
    
    def show_multiclass_dataset_popup(self, dataset_status):
        """Show detailed popup for existing multiclass dataset with progress tracking"""
        
        # Create custom dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("Multiclass Dataset Status")
        dialog.geometry("650x700")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        
        # Try to set grab, but handle gracefully if it fails
        try:
            dialog.grab_set()
        except tk.TclError:
            pass
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (650 // 2)
        y = (dialog.winfo_screenheight() // 2) - (700 // 2)
        dialog.geometry(f"650x700+{x}+{y}")
        
        # Result variable
        result = tk.BooleanVar(value=False)
        
        # Main frame
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text=self.texts["existing_multiclass_dataset_title"], 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 15))
        
        # Dataset info
        info_frame = ttk.LabelFrame(main_frame, text=self.texts["dataset_status"], padding="10")
        info_frame.pack(fill='x', pady=(0, 15))
        
        ttk.Label(info_frame, text=f"Dataset ID: {dataset_status['dataset_id']}", 
                 font=('Arial', 10, 'bold')).pack(anchor='w')
        ttk.Label(info_frame, text="Experiment Type: Multiclass Classification", 
                 font=('Arial', 10, 'bold'), foreground='purple').pack(anchor='w')
        ttk.Label(info_frame, text=f"Last Updated: {dataset_status['last_updated']}", 
                 font=('Arial', 9)).pack(anchor='w')
        
        # Multiclass information
        multiclass_info = dataset_status.get("multiclass_info", {})
        multiclass_frame = ttk.LabelFrame(main_frame, text=self.texts["multiclass_configuration_title"], padding="10")
        multiclass_frame.pack(fill='x', pady=(0, 15))
        
        strategy = multiclass_info.get("strategy", "UNKNOWN")
        n_classes = multiclass_info.get("n_classes", "?")
        class_names = multiclass_info.get("class_names", [])
        
        ttk.Label(multiclass_frame, text=f"Strategy: {strategy}", 
                 font=('Arial', 10, 'bold')).pack(anchor='w')
        ttk.Label(multiclass_frame, text=f"Number of Classes: {n_classes}", 
                 font=('Arial', 10)).pack(anchor='w')
        
        if class_names:
            classes_text = ", ".join(map(str, class_names))
            if len(classes_text) > 60:
                classes_text = classes_text[:60] + "..."
            ttk.Label(multiclass_frame, text=f"Classes: {classes_text}", 
                     font=('Arial', 9)).pack(anchor='w')
        
        # Progress info
        progress_frame = ttk.LabelFrame(main_frame, text=self.texts["dichotomy_progress_title"], padding="10")
        progress_frame.pack(fill='x', pady=(0, 15))
        
        completed_dichotomies = multiclass_info.get("completed_dichotomies", 0)
        total_dichotomies = multiclass_info.get("total_dichotomies", 0)
        in_progress_dichotomies = multiclass_info.get("in_progress_dichotomies", 0)
        
        # Overall progress
        overall_frame = ttk.Frame(progress_frame)
        overall_frame.pack(fill='x', pady=(0, 10))
        
        overall_status = dataset_status.get("overall_status", "unknown")
        if overall_status == "complete":
            status_text = self.texts["multiclass_status_complete"]
            status_color = "green"
        elif in_progress_dichotomies > 0:
            status_text = self.texts["multiclass_status_in_progress"]
            status_color = "blue"
        else:
            status_text = self.texts["multiclass_status_ready"]
            status_color = "orange"
        
        ttk.Label(overall_frame, text=f"Status: {status_text}", 
                 font=('Arial', 10, 'bold'), foreground=status_color).pack(anchor='w')
        ttk.Label(overall_frame, text=self.texts["multiclass_dichotomies_label"].format(
                     completed=completed_dichotomies, total=total_dichotomies), 
                 font=('Arial', 9)).pack(anchor='w', padx=(20, 0))
        
        if in_progress_dichotomies > 0:
            ttk.Label(overall_frame, text=self.texts["multiclass_in_progress_label"].format(
                         in_progress=in_progress_dichotomies), 
                     font=('Arial', 9)).pack(anchor='w', padx=(20, 0))
        
        # ETA information
        eta_formatted = dataset_status.get("eta_formatted")
        if eta_formatted and eta_formatted != "Unknown":
            using_estimate = dataset_status.get("using_estimate", True)
            if using_estimate:
                eta_text = self.texts["multiclass_eta_estimated"].format(eta=eta_formatted)
            else:
                eta_text = self.texts["multiclass_eta_label"].format(eta=eta_formatted)
            ttk.Label(overall_frame, text=eta_text, font=('Arial', 9), foreground='gray').pack(anchor='w', padx=(20, 0))
        
        # Best result info
        best_metric = dataset_status.get("best_metric_so_far")
        if best_metric:
            best_frame = ttk.LabelFrame(main_frame, text=self.texts["best_result_so_far_title"], padding="10")
            best_frame.pack(fill='x', pady=(0, 15))
            
            ttk.Label(best_frame, text=self.texts["best_metric_label"].format(metric=best_metric), 
                     font=('Arial', 10, 'bold')).pack(anchor='w')
            
            best_runner = dataset_status.get("best_runner")
            if best_runner:
                source_dichotomy = getattr(best_runner, 'source_dichotomy', 'Unknown')
                ttk.Label(best_frame, text=self.texts["from_dichotomy_label"].format(dichotomy=source_dichotomy), 
                         font=('Arial', 9)).pack(anchor='w')
        
        # Resume message
        if overall_status == "complete":
            message = self.texts["multiclass_complete_message"]
        else:
            remaining = total_dichotomies - completed_dichotomies
            message = self.texts["multiclass_continue_message"].format(remaining=remaining)
        
        message_label = ttk.Label(main_frame, text=message, font=('Arial', 10), 
                                 wraplength=600, justify='center')
        message_label.pack(pady=15)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(15, 0))
        
        def on_continue():
            result.set(True)
            dialog.destroy()
        
        def on_cancel():
            result.set(False)
            dialog.destroy()
        
        def on_delete_progress():
            # Close current dialog first
            dialog.destroy()
            
            # Show deletion confirmation and execute if confirmed
            data_source = self.data_source_var.get()
            csv_path = self.get_current_csv_path() if data_source == "real" else None
            
            self.delete_dataset_progress(data_source, csv_path, dataset_status)
            
            # Set result to False so the experiment doesn't start automatically
            result.set(False)
        
        # Delete progress button (left side) - Red with trash icon
        delete_button = tk.Button(button_frame, 
                                 text=f"🗑 {self.texts['delete_progress']}", 
                                 command=on_delete_progress,
                                 bg="#e53e3e",  # Red
                                 fg="white",
                                 font=("Arial", 9, "bold"),
                                 relief="raised",
                                 padx=10,
                                 pady=5,
                                 cursor="hand2")
        delete_button.pack(side='left')
        
        # Continue and Cancel buttons (right side)
        continue_button = tk.Button(button_frame, 
                                   text=f"✓ {self.texts['continue']}", 
                                   command=on_continue,
                                   bg="#4CAF50",  # Green
                                   fg="white",
                                   font=("Arial", 9, "bold"),
                                   relief="raised",
                                   padx=15,
                                   pady=5,
                                   cursor="hand2")
        continue_button.pack(side='right', padx=(10, 0))
        
        cancel_button = tk.Button(button_frame, 
                                 text=f"✗ {self.texts['cancel']}", 
                                 command=on_cancel,
                                 bg="#9e9e9e",  # Gray
                                 fg="white",
                                 font=("Arial", 9),
                                 relief="raised",
                                 padx=15,
                                 pady=5,
                                 cursor="hand2")
        cancel_button.pack(side='right')
        
        # Wait for dialog to close
        dialog.wait_window()
        
        return result.get()

    def delete_dataset_progress(self, data_source, csv_path, dataset_status):
        """Handle dataset progress deletion with confirmation"""
        # Show confirmation dialog
        title = self.texts["delete_progress_title"]
        message = self.texts["delete_progress_message"]
        
        confirm = messagebox.askyesno(title, message, icon='warning')
        
        if not confirm:
            return
        
        try:
            # Call the deletion method from experiment runner
            result = self.experiment_runner.delete_dataset_progress(data_source, csv_path)
            
            if result.get("success", False):
                # Show success message
                success_title = self.texts["progress_deleted"]
                success_message = self.texts["progress_deleted_message"]
                
                if result.get("deleted_items"):
                    success_message += f"\n\nDeleted items:\n• {chr(10).join(['• ' + item for item in result['deleted_items']])}"
                
                messagebox.showinfo(success_title, success_message)
                
                # Log the deletion
                self.add_log_message(f"Progress deleted for dataset {result.get('dataset_id', 'unknown')}: {result.get('message', 'Success')}")
                
            else:
                # Show error message
                error_title = self.texts["delete_failed"]
                error_message = result.get("message", "Unknown error occurred")
                messagebox.showerror(error_title, f"{error_message}")
                
                # Log the error
                self.add_log_message(f"Failed to delete progress: {error_message}")
                
        except Exception as e:
            # Handle unexpected errors
            error_title = self.texts["delete_failed"]
            error_message = f"Unexpected error: {str(e)}"
            messagebox.showerror(error_title, error_message)
            
            # Log the error
            self.add_log_message(f"Error deleting progress: {error_message}")

    def check_experiment_status(self):
        """Periodically check the status of the experiment process."""
        if not self.is_running:
            return

        # Get current progress information
        self.update_progress_display()

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

    def update_progress_display(self):
        """Update the progress bar and detail information"""
        try:
            # Get current dataset status with timing information
            data_source = self.data_source_var.get()
            csv_path = self.get_current_csv_path()
            current_level = self.level_var.get()
            
            progress_info = self.experiment_runner.get_dataset_status(data_source, csv_path, current_level)
            
            if progress_info and progress_info.get("status") == "existing":
                self.current_progress_info = progress_info
                self.update_progress_widgets(progress_info)
            else:
                # If no existing progress for current dataset, check for any running experiments
                if self.is_running:
                    running_info = self.experiment_runner.get_any_running_experiment_status()
                    if running_info:
                        self.current_progress_info = running_info
                        self.update_progress_widgets(running_info)
                    else:
                        self.progress_detail_var.set("Experiment running...")
            
        except Exception as e:
            # If progress update fails, show basic running info
            if self.is_running:
                self.progress_detail_var.set("Experiment running...")
    
    def get_current_csv_path(self):
        """Get the current CSV path based on user selection"""
        if self.data_source_var.get() == "real":
            if hasattr(self, 'user_csv_path') and self.user_csv_path:
                return self.user_csv_path
            elif hasattr(self, 'industry_csv_path') and self.industry_csv_path:
                return self.industry_csv_path
        return None
    
    def update_progress_widgets(self, progress_info):
        """Update progress bar and detail text based on progress information"""
        try:
            # Check if this is a multiclass experiment
            experiment_type = progress_info.get("experiment_type", "binary")
            
            if experiment_type == "multiclass":
                self.update_multiclass_progress_widgets(progress_info)
                return
            
            # Standard binary classification progress handling
            # Determine current stage and progress
            stage1_complete = progress_info.get("stage1_completed", False)
            current_stage = progress_info.get("current_stage", 1)
            
            # Get progress numbers for both stages
            stage1_progress = progress_info.get("stage1_progress", "0/0")
            stage2_progress = progress_info.get("stage2_progress", "0/0")
            
            # Determine which stage is currently active for progress bar
            if stage1_complete and current_stage == 2:
                # Stage 2 is active
                if "/" in stage2_progress:
                    current, total = map(int, stage2_progress.split("/"))
                else:
                    current, total = 0, 0
                stage_name = "Stage 2"
            else:
                # Stage 1 is active
                if "/" in stage1_progress:
                    current, total = map(int, stage1_progress.split("/"))
                else:
                    current, total = 0, 0
                stage_name = "Stage 1"
            
            # Update custom progress bar with both stages
            self.update_custom_progress_bar(stage1_progress, stage2_progress, stage1_complete, False)
            
            # Build detailed status text
            detail_parts = []
            
            # Add stage-specific ETA timing information
            eta_formatted = progress_info.get("eta_formatted")
            if eta_formatted and eta_formatted != "Unknown":
                using_estimate = progress_info.get("using_estimate", False)
                if using_estimate:
                    detail_parts.append(f"ETA for {stage_name}: ~{eta_formatted} (estimated)")
                else:
                    detail_parts.append(f"ETA for {stage_name}: {eta_formatted}")
            
            # Speed information removed as requested
            
            avg_time = progress_info.get("avg_time_per_run", 0)
            if avg_time > 0:
                using_estimate = progress_info.get("using_estimate", False)
                if using_estimate:
                    dataset_size = progress_info.get("dataset_size", 0)
                    detail_parts.append(f"Est: {avg_time:.1f}s/run (dataset: {dataset_size} rows)")
                else:
                    detail_parts.append(f"Avg: {avg_time:.1f}s/run")
            
            # Add best metric if available
            best_metric = progress_info.get("best_metric_so_far")
            if best_metric and best_metric != "N/A":
                detail_parts.append(f"Best: {best_metric:.3f}")
            
            # Update the detail text
            detail_text = " • ".join(detail_parts) if detail_parts else "Experiment running..."
            self.progress_detail_var.set(detail_text)
            
            # Update main status to show stage breakdown
            if self.is_running:
                # Build status text with stage breakdown
                if stage1_complete:
                    # Stage 1 is done, show both stages with emphasis on stage 2
                    status_text = f"Running Stage 1({stage1_progress}) Stage 2({stage2_progress})"
                else:
                    # Stage 1 is still running
                    status_text = f"Running Stage 1({stage1_progress}) Stage 2({stage2_progress})"
                
                self.status_var.set(status_text)
        
        except Exception as e:
            # If widget update fails, show basic info
            self.progress_detail_var.set("Experiment running...")
            self.update_custom_progress_bar("0/0", "0/0", False, False)

    def update_multiclass_progress_widgets(self, progress_info):
        """Update progress widgets specifically for multiclass experiments"""
        try:
            multiclass_info = progress_info.get("multiclass_info", {})
            
            # Extract multiclass-specific progress
            completed_dichotomies = multiclass_info.get("completed_dichotomies", 0)
            total_dichotomies = multiclass_info.get("total_dichotomies", 0)
            strategy = multiclass_info.get("strategy", "UNKNOWN")
            n_classes = multiclass_info.get("n_classes", "?")
            
            # Create progress string that matches expected format
            dichotomy_progress = f"{completed_dichotomies}/{total_dichotomies}"
            
            # For multiclass, we treat dichotomies as "Stage 1" and show no Stage 2
            stage1_complete = completed_dichotomies >= total_dichotomies
            
            # Update custom progress bar - use dichotomy progress as Stage 1, empty Stage 2
            self.update_custom_progress_bar(dichotomy_progress, "0/0", stage1_complete, stage1_complete)
            
            # Build detailed status text for multiclass
            detail_parts = []
            
            # Add strategy and class information
            detail_parts.append(f"{strategy} • {n_classes} classes")
            
            # Add ETA timing information
            eta_formatted = progress_info.get("eta_formatted")
            if eta_formatted and eta_formatted != "Unknown":
                using_estimate = progress_info.get("using_estimate", False)
                if using_estimate:
                    detail_parts.append(f"ETA: ~{eta_formatted} (estimated)")
                else:
                    detail_parts.append(f"ETA: {eta_formatted}")
            
            # Add timing information for dichotomies
            avg_time = progress_info.get("avg_time_per_run", 0)
            if avg_time > 0:
                using_estimate = progress_info.get("using_estimate", False)
                if avg_time > 3600:  # More than 1 hour
                    hours = int(avg_time / 3600)
                    minutes = int((avg_time % 3600) / 60)
                    time_str = f"{hours}h{minutes}m" if hours > 0 else f"{minutes}m"
                else:
                    minutes = int(avg_time / 60)
                    time_str = f"{minutes}m" if minutes > 0 else f"{avg_time:.0f}s"
                
                if using_estimate:
                    detail_parts.append(f"Est: {time_str}/dichotomy")
                else:
                    detail_parts.append(f"Avg: {time_str}/dichotomy")
            
            # Add best metric if available
            best_metric = progress_info.get("best_metric_so_far")
            if best_metric and best_metric != "N/A":
                detail_parts.append(f"Best: {best_metric:.3f}")
            
            # Update the detail text
            detail_text = " • ".join(detail_parts) if detail_parts else "Multiclass experiment running..."
            self.progress_detail_var.set(detail_text)
            
            # Update main status to show multiclass progress
            if self.is_running:
                if stage1_complete:
                    status_text = f"Multiclass Complete ({dichotomy_progress} dichotomies)"
                else:
                    # Check if there's a current dichotomy being processed
                    current_dichotomy_index = multiclass_info.get("current_dichotomy_index")
                    if current_dichotomy_index:
                        status_text = f"Processing Dichotomy {current_dichotomy_index}/{total_dichotomies} • {completed_dichotomies} completed"
                    else:
                        status_text = f"Running Multiclass ({dichotomy_progress} dichotomies)"
                
                self.status_var.set(status_text)
        
        except Exception as e:
            # If multiclass widget update fails, show basic info
            self.progress_detail_var.set("Multiclass experiment running...")
            self.update_custom_progress_bar("0/0", "0/0", False, False)

    def experiment_completed(self, message):
        """Handle experiment completion"""
        self.is_running = False
        self.start_button.configure(state='normal', bg="#4CAF50")   # Green when enabled
        self.stop_button.configure(state='disabled', bg="#cccccc") # Gray when disabled
        self.status_var.set(self.texts["completed"])
        
        # Reset progress display - show completed state
        # For completed experiments, we'll show both stages as complete
        self.update_custom_progress_bar("100/100", "100/100", True, True)  # Visual completion with green Stage 2
        self.progress_detail_var.set("Experiment completed successfully")
        
        self.add_log_message(message)
        
        # Add completion separator with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.add_log_message("-" * 60)
        self.add_log_message(f"EXPERIMENT COMPLETED - {timestamp}")
        self.add_log_message("-" * 60)
        
        # Show success message first
        messagebox.showinfo(self.texts["success"], message)
        
        # Automatically generate and show report popup
        self.generate_and_show_report()
    
    def experiment_error(self, error_msg):
        """Handle experiment error"""
        self.is_running = False
        self.start_button.configure(state='normal', bg="#4CAF50")   # Green when enabled
        self.stop_button.configure(state='disabled', bg="#cccccc") # Gray when disabled
        self.status_var.set(self.texts["error"])
        
        # Reset progress display
        self.update_custom_progress_bar("0/0", "0/0", False, False)
        self.progress_detail_var.set("Experiment terminated with error")
        
        self.add_log_message(f"ERROR: {error_msg}")
        
        # Add error separator with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.add_log_message("-" * 60)
        self.add_log_message(f"EXPERIMENT ERROR - {timestamp}")
        self.add_log_message("-" * 60)
        
        messagebox.showerror(self.texts["error"], error_msg)
    
    def generate_and_show_report(self):
        """Automatically generate and show experiment report popup"""
        try:
            # Add log message about report generation
            self.add_log_message("Generating experiment report...")
            
            # Get current experiment data
            data_source = self.data_source_var.get()
            csv_path = self.get_current_csv_path()
            current_level = self.level_var.get()
            
            # Generate report through experiment runner
            self.experiment_runner.generate_experiment_report(
                data_source=data_source,
                csv_path=csv_path,
                current_level=current_level,
                gui_root=self.root,
                language=self.language
            )
            
            self.add_log_message("Report generation completed")
            
        except Exception as e:
            self.add_log_message(f"Error generating report: {str(e)}")
            # Don't show error popup to avoid interrupting the completion flow
            # Just log the error so the experiment completion isn't disrupted
    
    def stop_experiment(self):
        """Stop the current experiment"""
        if self.is_running:
            self.add_log_message(self.texts["stopping_experiment"])
            try:
                result = self.experiment_runner.stop_experiment()
                self.add_log_message(result)
            except Exception as e:
                self.add_log_message(f"Error stopping experiment: {str(e)}")
            
            # Add end separator with timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.add_log_message("-" * 60)
            self.add_log_message(f"EXPERIMENT ENDED - {timestamp}")
            self.add_log_message("-" * 60)
            
            self.is_running = False
            self.start_button.configure(state='normal', bg="#4CAF50")   # Green when enabled
            self.stop_button.configure(state='disabled', bg="#cccccc") # Gray when disabled
            self.status_var.set(self.texts["stopped"])
            
            # Reset progress display
            self.update_custom_progress_bar("0/0", "0/0", False, False)
            self.progress_detail_var.set("Experiment stopped by user")
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
                config_window.title(self.texts["config_window_title"])
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
            if messagebox.askokcancel(self.texts["quit_dialog_title"], self.texts["quit_confirm"]):
                self.stop_experiment()
                self.root.destroy()
        else:
            self.root.destroy()







