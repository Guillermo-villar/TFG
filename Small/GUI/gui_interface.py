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
        
        # GUI variables
        self.level_var = tk.IntVar(value=1)
        self.status_var = tk.StringVar(value=self.texts["ready"])
        self.study_mode_var = tk.StringVar(value="full_study")
        self.data_source_var = tk.StringVar(value="synthetic")
        self.csv_path_var = tk.StringVar(value="")
        self.csv_full_path_display_var = tk.StringVar(value="")
        self.popup_log_text = None  # For the popup window
        
        self.full_csv_path = ""
        self.csv_file_map = {}
        
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
        """Called when user selects a CSV file from the dropdown"""
        selected_display_name = self.csv_path_var.get()
        
        # Get the full path from the map
        self.full_csv_path = self.csv_file_map.get(selected_display_name, "")
        self.csv_full_path_display_var.set(self.full_csv_path)

        if self.full_csv_path and os.path.exists(self.full_csv_path):
            # Automatically switch to real data mode when industry dataset is selected
            self.data_source_var.set("real")
            self.toggle_csv_options()  # Update UI state
            
            # Show friendly name for industry datasets
            self.add_log_message(f"Selected industry test dataset: {selected_display_name}")
        else:
            # This shouldn't happen with industry datasets, but handle gracefully
            self.add_log_message(f"Warning: Selected file does not exist: {self.full_csv_path}")

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
                    # Set to first industry dataset
                    first_display_name = display_names[0]
                    self.csv_path_var.set(first_display_name)
                    self.full_csv_path = self.csv_file_map[first_display_name]
                    self.csv_full_path_display_var.set(self.full_csv_path)
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
                
                # Create processed filename
                base_name = os.path.splitext(os.path.basename(filepath))[0]
                processed_path = os.path.join(os.path.dirname(filepath), f"{base_name}_processed.csv")
                
                # Show preprocessing dialog to user
                proceed = messagebox.askyesno(
                    "CSV Preprocessing", 
                    f"The selected CSV file will be analyzed and preprocessed if needed.\n\n"
                    f"Original file: {os.path.basename(filepath)}\n"
                    f"Processed file: {os.path.basename(processed_path)}\n\n"
                    f"Detected columns: {len(analysis['columns'])}\n"
                    f"Binary columns found: {len(analysis['binary_columns'])}\n"
                    f"Text columns found: {len(analysis.get('text_columns', []))}\n"
                    f"Categorical columns found: {len(analysis.get('categorical_columns', []))}\n\n"
                    f"Do you want to proceed with preprocessing?"
                )
                
                if proceed:
                    # Process the CSV file with GUI interaction
                    result = preprocessor.process_csv_file(filepath, processed_path)
                    
                    if result['success']:
                        final_path = result['output_path']
                        self.add_log_message(f"CSV preprocessing completed: {os.path.basename(final_path)}")
                        
                        # Set the processed file as current selection (don't add to dropdown)
                        self.csv_path_var.set(final_path)
                        
                        # Show preprocessing summary
                        log_summary = "\n".join(result['processing_log'])
                        if log_summary:
                            messagebox.showinfo("Preprocessing Complete", 
                                              f"Preprocessing completed successfully!\n\nSummary:\n{log_summary}")
                    else:
                        # Preprocessing failed
                        error_msg = result.get('error', 'Unknown preprocessing error')
                        messagebox.showerror("Preprocessing Failed", 
                                           f"Preprocessing failed: {error_msg}\n\nUsing original file.")
                        # Set original file as current selection (don't add to dropdown) 
                        self.csv_path_var.set(filepath)
                else:
                    # User declined preprocessing, set original file as current selection (don't add to dropdown)
                    self.csv_path_var.set(filepath)
                    
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
                    
                messagebox.showerror("Preprocessing Error", 
                                   f"Error during preprocessing:\n{error_msg}\n\nUsing original file.")
                # Set original file as current selection (don't add to dropdown)
                self.csv_path_var.set(filepath)
    
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
        
        # Status
        status_frame = ttk.Frame(control_frame)
        status_frame.grid(row=0, column=1, sticky=tk.E)
        
        ttk.Label(status_frame, text=self.texts["status"]).pack(side='left', padx=(0, 5))
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                font=('Arial', 10, 'bold'))
        status_label.pack(side='left')

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
        csv_path = self.full_csv_path if data_source == "real" else None
        
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
        
        # Add separator with timestamp instead of clearing log to stack experiments
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.add_log_message("=" * 60)
        self.add_log_message(f"NEW EXPERIMENT - {timestamp}")
        self.add_log_message("=" * 60)
        self.add_log_message(self.texts["starting_experiment"].format(level, study_mode))
        if data_source == 'real':
            self.add_log_message(self.texts["using_real_data"].format(csv_path))
        else:
            self.add_log_message(self.texts["using_synthetic_data"])

        try:
            result = self.experiment_runner.run_experiment(
                level=level, 
                study_mode=study_mode,
                data_source=data_source,
                csv_path=csv_path
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
        dialog.grab_set()
        
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
            csv_path = self.csv_path_var.get() if data_source == "real" else None
            
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
                                 text="✗ Cancel", 
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
        
        # Create custom dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title(self.texts["dataset_status"])
        dialog.geometry("600x650")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
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
            csv_path = self.csv_path_var.get() if data_source == "real" else None
            
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
                                 text="✗ Cancel", 
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
        self.start_button.configure(state='normal', bg="#4CAF50")   # Green when enabled
        self.stop_button.configure(state='disabled', bg="#cccccc") # Gray when disabled
        self.status_var.set(self.texts["completed"])
        
        self.add_log_message(message)
        
        # Add completion separator with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.add_log_message("-" * 60)
        self.add_log_message(f"EXPERIMENT COMPLETED - {timestamp}")
        self.add_log_message("-" * 60)
        
        messagebox.showinfo(self.texts["success"], message)
    
    def experiment_error(self, error_msg):
        """Handle experiment error"""
        self.is_running = False
        self.start_button.configure(state='normal', bg="#4CAF50")   # Green when enabled
        self.stop_button.configure(state='disabled', bg="#cccccc") # Gray when disabled
        self.status_var.set(self.texts["error"])
        
        self.add_log_message(f"ERROR: {error_msg}")
        
        # Add error separator with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.add_log_message("-" * 60)
        self.add_log_message(f"EXPERIMENT ERROR - {timestamp}")
        self.add_log_message("-" * 60)
        
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
            
            # Add end separator with timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.add_log_message("-" * 60)
            self.add_log_message(f"EXPERIMENT ENDED - {timestamp}")
            self.add_log_message("-" * 60)
            
            self.is_running = False
            self.start_button.configure(state='normal', bg="#4CAF50")   # Green when enabled
            self.stop_button.configure(state='disabled', bg="#cccccc") # Gray when disabled
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







