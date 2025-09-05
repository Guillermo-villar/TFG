import tkinter as tk
from tkinter import ttk, messagebox
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
