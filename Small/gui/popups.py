import tkinter as tk
from tkinter import ttk, messagebox

class PopupManager:
    def __init__(self, gui):
        self.gui = gui
        self.root = gui.root
        self.texts = gui.texts

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
            data_source = self.gui.data_source_var.get()
            csv_path = self.gui.get_current_csv_path() if data_source == "real" else None
            
            self.gui.delete_dataset_progress(data_source, csv_path, dataset_status)
            
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
            data_source = self.gui.data_source_var.get()
            csv_path = self.gui.get_current_csv_path() if data_source == "real" else None
            
            self.gui.delete_dataset_progress(data_source, csv_path, dataset_status)
            
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
            best_metric_frame = ttk.LabelFrame(main_frame, text=self.texts["best_result_so_far_title"], padding="10")
            best_metric_frame.pack(fill='x', pady=(0, 15))
            
            metric_name = best_metric.get("metric_name", "Accuracy")
            metric_value = best_metric.get("value", 0)
            
            ttk.Label(best_metric_frame, text=f"{metric_name}: {metric_value:.4f}", 
                     font=('Arial', 10, 'bold')).pack(anchor='w')
        
        # Resume message
        if overall_status == "complete":
            message = self.texts["multiclass_study_complete_message"]
        else:
            message = self.texts["multiclass_resume_message"]
        
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
            data_source = self.gui.data_source_var.get()
            csv_path = self.gui.get_current_csv_path() if data_source == "real" else None
            
            self.gui.delete_dataset_progress(data_source, csv_path, dataset_status)
            
            # Set result to False so the experiment doesn't start automatically
            result.set(False)
        
        # Delete progress button (left side)
        delete_button = tk.Button(button_frame, 
                                 text=f"🗑 {self.texts['delete_progress']}", 
                                 command=on_delete_progress,
                                 bg="#e53e3e", fg="white", font=("Arial", 9, "bold"),
                                 relief="raised", padx=10, pady=5, cursor="hand2")
        delete_button.pack(side='left')
        
        # Continue and Cancel buttons (right side)
        continue_button = tk.Button(button_frame, 
                                   text=f"✓ {self.texts['continue']}", 
                                   command=on_continue,
                                   bg="#4CAF50", fg="white", font=("Arial", 9, "bold"),
                                   relief="raised", padx=15, pady=5, cursor="hand2")
        continue_button.pack(side='right', padx=(10, 0))
        
        cancel_button = tk.Button(button_frame, 
                                 text=f"✗ {self.texts['cancel']}", 
                                 command=on_cancel,
                                 bg="#9e9e9e", fg="white", font=("Arial", 9),
                                 relief="raised", padx=15, pady=5, cursor="hand2")
        cancel_button.pack(side='right')
        
        # Wait for dialog to close
        dialog.wait_window()
        
        return result.get()
