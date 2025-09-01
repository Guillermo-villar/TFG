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
    def __init__(self, parent, pdf_path, save_callback=None):
        self.parent = parent
        self.pdf_path = pdf_path
        self.save_callback = save_callback
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
        self.popup.title("PDF Report Preview")
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

def parse_experiment_folder(experiment_folder):
    """Parse experiment folder to extract data for report generation"""
    import json
    import csv

    dataset_name = os.path.basename(experiment_folder)

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
        "dataset_name": dataset_name,
        "is_multiclass": is_multiclass,
        "strategy": strategy,
        "dichotomy_mapping": dichotomy_mapping,
        "class_distribution": class_distribution,
        "imbalance_ratio": imbalance_ratio,
        "metrics": metrics,
        "best_model": best_model,
    }


def show_report_popup(experiment_path: str, language: str = "en", parent=None) -> None:
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

    PDFPreviewPopup(root, temp_pdf_path, save_callback=_save)

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
