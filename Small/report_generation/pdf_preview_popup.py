"""

source /home/ubuntu/Downloads/Code/venv/bin/activate && cd /home/ubuntu/Downloads/Code/NeonGenesis/Small/report_generation && python pdf_preview_popup.py


pdf_preview_popup.py
Show a PDF preview in a Tkinter popup using pdf2image and Pillow.
"""
import os
from tkinter import Toplevel, Button, Label, LEFT, RIGHT, BOTH, Frame, X, Canvas, Scrollbar, VERTICAL
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from pdf2image import convert_from_path

class PDFPreviewPopup:
    def __init__(self, parent, pdf_path, save_callback=None, experiment_data=None, experiment_runner=None, title="PDF Report Preview"):
        self.parent = parent
        self.pdf_path = pdf_path
        self.save_callback = save_callback
        self.experiment_data = experiment_data  # Added to check if multiclass
        self.experiment_runner = experiment_runner  # Added for reconstitution callback
        self.title = title  # Window title
        self.images = []
        self.tk_images = []
        self.current_page = 0
        self._load_images()
        self._show_popup()

    def _load_images(self):
        # Convert PDF to images (one per page)
        self.images = convert_from_path(self.pdf_path, dpi=120)
        
        # Resize images to fit in popup (keeping aspect ratio)
        max_width = 650
        max_height = 800  # Increased height for better viewing
        
        self.tk_images = []
        for img in self.images:
            # Calculate scaling factor to fit within max dimensions
            width_ratio = max_width / img.width
            height_ratio = max_height / img.height
            scale_factor = min(width_ratio, height_ratio)
            
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.tk_images.append(ImageTk.PhotoImage(resized_img))

    def _show_popup(self):
        self.popup = Toplevel(self.parent)
        self.popup.title(self.title)
        self.popup.geometry("700x900")
        self.popup.configure(bg="#ffffff")
        self.popup.resizable(False, False)

        # Title label
        title_label = Label(self.popup, text="Preview", font=("Segoe UI", 16, "bold"), 
                           bg="#ffffff", fg="#333333")
        title_label.pack(pady=(15, 10))

        # PDF preview area with scrollable canvas
        preview_frame = Frame(self.popup, bg="#f0f0f0", relief="solid", bd=1, height=750)
        preview_frame.pack(padx=20, pady=(0, 10), fill=X)
        preview_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        # Create canvas and scrollbar
        self.canvas = Canvas(preview_frame, bg="#ffffff", highlightthickness=0)
        scrollbar = Scrollbar(preview_frame, orient=VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = Frame(self.canvas, bg="#ffffff")
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill="y")
        
        # Add all PDF pages to the scrollable frame
        for i, img in enumerate(self.tk_images):
            if i > 0:  # Add spacing between pages
                spacer = Label(self.scrollable_frame, text=f"--- Page {i+1} ---", 
                              font=("Segoe UI", 10), bg="#ffffff", fg="#999999")
                spacer.pack(pady=(20, 10))
            
            img_label = Label(self.scrollable_frame, image=img, bg="#ffffff", relief="flat")
            img_label.pack(pady=10)
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.canvas.bind("<MouseWheel>", _on_mousewheel)  # Windows
        self.canvas.bind("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))  # Linux
        self.canvas.bind("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))   # Linux

        # Bottom buttons - fixed at bottom
        button_frame = Frame(self.popup, bg="#ffffff", height=60)
        button_frame.pack(fill="x", pady=(0, 15), padx=20)
        button_frame.pack_propagate(False)
        
        # Save button (left)
        save_btn = Button(button_frame, text="Save PDF", font=("Segoe UI", 11), 
                         bg="#007acc", fg="white", relief="flat", padx=20, pady=8,
                         command=self.save_pdf)
        save_btn.pack(side=LEFT, pady=10)
        
        # Reconstitute Models button (center) - only for multiclass experiments
        if self.experiment_data and self.experiment_data.get('is_multiclass', False):
            self.reconstitute_btn = Button(button_frame, text="Reconstitute Models", font=("Segoe UI", 11), 
                                    bg="#FFC107", fg="black", relief="flat", padx=15, pady=8,
                                    command=self.reconstitute_models)
            self.reconstitute_btn.pack(side=LEFT, padx=(10, 0), pady=10)
        
        # Close button (right)
        close_btn = Button(button_frame, text="Close", font=("Segoe UI", 11), 
                          bg="#6c757d", fg="white", relief="flat", padx=20, pady=8,
                          command=self.popup.destroy)
        close_btn.pack(side=RIGHT, pady=10)

    def save_pdf(self):
        file_path = filedialog.asksaveasfilename(parent=self.popup, defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if file_path:
            if self.save_callback:
                self.save_callback(file_path)
            else:
                # Default: just copy the temp PDF
                import shutil
                shutil.copy(self.pdf_path, file_path)
            messagebox.showinfo("PDF Saved", f"Report saved to {file_path}", parent=self.popup)

    def reconstitute_models(self):
        """Handle the Reconstitute Models button click with loading state"""
        if not self.experiment_data or not self.experiment_runner:
            messagebox.showerror("Error", "Cannot reconstitute models: Missing experiment data or runner.", parent=self.popup)
            return
        
        try:
            # Extract dataset ID from the experiment data
            dataset_id = self.experiment_data.get('dataset_name')
            if not dataset_id:
                messagebox.showerror("Error", "Cannot find dataset ID for reconstitution.", parent=self.popup)
                return
            
            # Show confirmation dialog
            result = messagebox.askyesno(
                "Reconstitute Models", 
                f"This will reconstitute the multiclass model from the best dichotomy configurations.\n\n"
                f"Dataset: {dataset_id}\n"
                f"Strategy: {self.experiment_data.get('strategy', 'Unknown')}\n"
                f"Dichotomies: {len(self.experiment_data.get('metrics', {}))}\n\n"
                f"This process may take several minutes. Continue?",
                parent=self.popup
            )
            
            if result:
                # Start the reconstitution process in the background
                self._start_reconstitution_process(dataset_id)
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start reconstitution: {str(e)}", parent=self.popup)

    def _start_reconstitution_process(self, dataset_id):
        """Start the reconstitution process with loading UI"""
        import threading
        
        # Change button to loading state
        self.reconstitute_btn.config(
            text="Reconstituting... ⟳", 
            state="disabled", 
            bg="#FFA000"
        )
        
        # Set cursor on the popup window instead
        try:
            self.popup.config(cursor="watch")
        except:
            # Fallback if cursor fails
            pass
        
        # Show loading dialog
        self._create_loading_dialog()
        
        # Start reconstitution in background thread
        def reconstitution_worker():
            try:
                reconstitution_result = self.experiment_runner.reconstitute_multiclass_models(dataset_id)
                
                # Schedule UI update on main thread
                self.popup.after(100, lambda: self._on_reconstitution_complete(reconstitution_result, dataset_id))
                
            except Exception as e:
                # Schedule error handling on main thread
                self.popup.after(100, lambda: self._on_reconstitution_error(str(e)))
        
        # Start background thread
        thread = threading.Thread(target=reconstitution_worker, daemon=True)
        thread.start()

    def _create_loading_dialog(self):
        """Create a loading dialog with progress animation"""
        self.loading_window = Toplevel(self.popup)
        self.loading_window.title("Processing")
        self.loading_window.geometry("450x280")
        self.loading_window.resizable(False, False)
        self.loading_window.configure(bg="#f8f9fa")
        
        # Center the loading window
        self.loading_window.transient(self.popup)
        self.loading_window.grab_set()
        
        # Loading content
        content_frame = Frame(self.loading_window, bg="#f8f9fa")
        content_frame.pack(expand=True, fill="both", padx=30, pady=30)
        
        # Animated loading text
        self.loading_label = Label(content_frame, 
                                  text="Reconstituting multiclass models...", 
                                  font=("Segoe UI", 14),
                                  bg="#f8f9fa", fg="#333333")
        self.loading_label.pack(pady=(20, 10))
        
        # Progress steps
        steps_frame = Frame(content_frame, bg="#f8f9fa")
        steps_frame.pack(pady=10)
        
        self.step_labels = []
        steps = [
            "Loading experiment configuration",
            "Training optimized models", 
            "Running multiple simulations",
            "Calculating performance metrics",
            "Generating report"
        ]
        
        for i, step in enumerate(steps):
            label = Label(steps_frame, text=f"• {step}", 
                         font=("Segoe UI", 10), 
                         bg="#f8f9fa", fg="#666666",
                         anchor="w")
            label.pack(fill="x", pady=2)
            self.step_labels.append(label)
        
        # Start animation
        self._animate_loading()

    def _animate_loading(self):
        """Animate the loading text"""
        if hasattr(self, 'loading_window') and self.loading_window.winfo_exists():
            current_text = self.loading_label.cget("text")
            if current_text.endswith("..."):
                new_text = current_text[:-3] + "."
            elif current_text.endswith(".."):
                new_text = current_text + "."
            elif current_text.endswith("."):
                new_text = current_text + "."
            else:
                new_text = current_text + "."
            
            self.loading_label.config(text=new_text)
            
            # Continue animation
            self.popup.after(500, self._animate_loading)

    def _on_reconstitution_complete(self, result, dataset_id):
        """Handle successful reconstitution completion"""
        # Close loading dialog
        if hasattr(self, 'loading_window'):
            self.loading_window.destroy()
        
        # Reset button and cursor
        self.reconstitute_btn.config(
            text="Reconstitute Models", 
            state="normal", 
            bg="#FFC107"
        )
        
        # Reset cursor
        try:
            self.popup.config(cursor="")
        except:
            pass
        
        if result.get('success', False):
            # Show success message and offer to view results
            msg_result = messagebox.askyesno(
                "Reconstitution Complete",
                f"Model reconstitution completed successfully!\n\n"
                f"{result.get('message', '')}\n\n"
                f"Would you like to view the detailed reconstitution report?",
                parent=self.popup
            )
            
            if msg_result:
                # Generate and show reconstitution PDF report
                self._generate_reconstitution_report(dataset_id, result.get('results', {}))
                
        else:
            messagebox.showerror("Error", 
                                f"Reconstitution failed:\n{result.get('message', 'Unknown error')}", 
                                parent=self.popup)

    def _on_reconstitution_error(self, error_message):
        """Handle reconstitution error"""
        # Close loading dialog
        if hasattr(self, 'loading_window'):
            self.loading_window.destroy()
        
        # Reset button and cursor
        self.reconstitute_btn.config(
            text="Reconstitute Models", 
            state="normal", 
            bg="#FFC107"
        )
        
        # Reset cursor
        try:
            self.popup.config(cursor="")
        except:
            pass
        
        messagebox.showerror("Error", f"Reconstitution failed: {error_message}", parent=self.popup)

    def _generate_reconstitution_report(self, dataset_id, results):
        """Generate a specialized PDF report for reconstitution results"""
        try:
            from report_generation.multiclass_report_generator import MulticlassReportGenerator
            
            # Generate the multiclass reconstitution report
            report_generator = MulticlassReportGenerator(dataset_id, results)
            pdf_path = report_generator.generate_report()
            
            # Show the reconstitution report in a new popup
            if pdf_path and os.path.exists(pdf_path):
                reconstitution_popup = PDFPreviewPopup(
                    self.popup, 
                    pdf_path, 
                    title=f"Multiclass Reconstitution Report - {dataset_id}",
                    experiment_data=None,  # No reconstitute button for reconstitution reports
                    experiment_runner=None
                )
            else:
                messagebox.showerror("Error", "Failed to generate reconstitution report", parent=self.popup)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {str(e)}", parent=self.popup)

def parse_experiment_folder(experiment_folder):
    """Parse experiment folder to extract data for report generation"""
    import json
    import csv

    # Try to get the original dataset name from the actual CSV file instead of the hash-based folder name
    dataset_name = os.path.basename(experiment_folder)
    original_dataset_name = dataset_name  # fallback to folder name
    
    # Look for the original CSV file to get the real filename
    for f in os.listdir(experiment_folder):
        if f.endswith('.csv') and not ('dichotomy' in f or 'processed' in f):
            # Use the original filename without extension as the dataset name
            original_dataset_name = os.path.splitext(f)[0]
            break
        elif f.endswith('_original.csv'):
            # Fallback to old naming convention
            original_dataset_name = os.path.splitext(f)[0].replace('_original', '')
            break

    # Detect multiclass or binary by counting dichotomy folders
    dichotomies = [
        d
        for d in os.listdir(experiment_folder)
        if d.startswith('dichotomy_') and os.path.isdir(os.path.join(experiment_folder, d))
    ]
    is_multiclass = len(dichotomies) > 1

    # Get strategy from multiclass progress JSON
    strategy = None
    dichotomy_mapping = {}
    if is_multiclass:
        progress_file = os.path.join(experiment_folder, f"multiclass_progress_{dataset_name}.json")
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                    strategy = progress_data.get('strategy', 'unknown')

                    # Create dichotomy mapping for OVO/OVA strategy
                    if 'class_names' in progress_data:
                        class_names = progress_data['class_names']
                        if strategy == 'ovo':
                            dichotomy_counter = 1
                            for i in range(len(class_names)):
                                for j in range(i + 1, len(class_names)):
                                    dichotomy_mapping[f'dichotomy_{dichotomy_counter:02d}'] = {
                                        'class_1': class_names[i],
                                        'class_2': class_names[j],
                                        'description': f'{class_names[i]} vs {class_names[j]}'
                                    }
                                    dichotomy_counter += 1
                        elif strategy == 'ova':
                            for i, class_name in enumerate(class_names, 1):
                                other_classes = [c for c in class_names if c != class_name]
                                dichotomy_mapping[f'dichotomy_{i:02d}'] = {
                                    'class_1': class_name,
                                    'class_2': f"Not {class_name}",
                                    'description': f'{class_name} vs {" + ".join(other_classes)}'
                                }
            except Exception as e:
                print(f"Error reading progress file: {e}")

    # Get class distribution from *_original.csv
    class_distribution = {}
    for f in os.listdir(experiment_folder):
        if f.endswith('_original.csv'):
            csv_path = os.path.join(experiment_folder, f)
            try:
                with open(csv_path, 'r') as file:
                    reader = csv.reader(file)
                    header = next(reader)
                    label_idx = -1
                    for i, col in enumerate(header):
                        if col.lower() in ['label', 'class', 'target']:
                            label_idx = i
                            break
                    if label_idx == -1:
                        label_idx = -1  # fallback: last column

                    class_counts = {}
                    for row in reader:
                        if row:  # skip empty rows
                            label = row[label_idx]
                            class_counts[label] = class_counts.get(label, 0) + 1
                    class_distribution = class_counts
            except Exception as e:
                print(f"Error reading CSV: {e}")
            break

    # Calculate imbalance ratio
    imbalance_ratio = None
    if class_distribution and len(class_distribution) > 1:
        max_c = max(class_distribution.values())
        min_c = min(class_distribution.values())
        imbalance_ratio = round(max_c / min_c, 2) if min_c > 0 else 'inf'

    # Parse best models and metrics
    metrics = {}
    best_model = {}
    for d in dichotomies:
        d_path = os.path.join(experiment_folder, d)

        # Parse best runner
        best_runner_path = os.path.join(d_path, f"best_runner_{d}.json")
        if os.path.exists(best_runner_path):
            try:
                with open(best_runner_path) as f:
                    best_model[d] = json.load(f)
            except Exception as e:
                print(f"Error reading {best_runner_path}: {e}")

        # Find latest test results
        runs_dir = os.path.join(d_path, 'runs')
        if os.path.isdir(runs_dir):
            run_folders = sorted([
                r for r in os.listdir(runs_dir)
                if os.path.isdir(os.path.join(runs_dir, r))
            ], reverse=True)
            for run in run_folders:
                test_results = os.path.join(runs_dir, run, 'test_results.csv')
                if os.path.exists(test_results):
                    try:
                        with open(test_results, 'r') as f:
                            reader = csv.reader(f)
                            header = next(reader)
                            row = next(reader)
                            metrics[d] = {header[i]: row[i] for i in range(len(header))}
                    except Exception as e:
                        print(f"Error reading {test_results}: {e}")
                    break

    return {
        "dataset_name": original_dataset_name,  # Use the original filename instead of hash
        "is_multiclass": is_multiclass,
        "strategy": strategy,
        "dichotomy_mapping": dichotomy_mapping,
        "class_distribution": class_distribution,
        "imbalance_ratio": imbalance_ratio,
        "metrics": metrics,
        "best_model": best_model,
    }


def show_report_popup(experiment_path: str, language: str = "en", parent=None, experiment_runner=None) -> None:
    """Generate the PDF report and show a preview popup within this project."""
    import tkinter as tk
    try:
        from .report_html_generator import ReportHTMLGenerator
        from .pdf_report_generator import PDFReportGenerator
    except Exception:
        # Fallback when executed as a script directly
        from report_html_generator import ReportHTMLGenerator
        from pdf_report_generator import PDFReportGenerator

    if not os.path.isdir(experiment_path):
        raise FileNotFoundError(f"Experiment path not found: {experiment_path}")

    # Parse experiment data and build HTML
    experiment_data = parse_experiment_folder(experiment_path)
    html_content = ReportHTMLGenerator(experiment_data, language=language).render_html()

    # Generate PDF to temp file
    pdf_generator = PDFReportGenerator(html_content)
    temp_pdf_path = pdf_generator.save_temp_pdf()

    # Show preview
    created_root = False
    root = parent
    if root is None:
        root = tk.Tk()
        root.withdraw()
        created_root = True

    def _save(file_path: str):
        pdf_generator.save_pdf(file_path)

    PDFPreviewPopup(root, temp_pdf_path, save_callback=_save, experiment_data=experiment_data, experiment_runner=experiment_runner)

    if created_root:
        try:
            root.mainloop()
        finally:
            try:
                os.unlink(temp_pdf_path)
            except Exception:
                pass


if __name__ == "__main__":
    # Minimal CLI for quick local preview (still within project context)
    import argparse
    parser = argparse.ArgumentParser(description="Preview experiment PDF report")
    parser.add_argument("experiment_path", help="Path to experiment folder")
    parser.add_argument("--lang", default="en", help="Language code, e.g. en or es")
    args = parser.parse_args()
    show_report_popup(args.experiment_path, language=args.lang)
