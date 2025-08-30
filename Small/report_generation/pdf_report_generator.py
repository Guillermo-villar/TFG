"""
pdf_report_generator.py
Generates a PDF from HTML using WeasyPrint.
"""
from weasyprint import HTML
import tempfile
import os

class PDFReportGenerator:
    def __init__(self, html_content):
        self.html_content = html_content

    def save_pdf(self, file_path):
        HTML(string=self.html_content).write_pdf(file_path)

    def get_pdf_bytes(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            HTML(string=self.html_content).write_pdf(tmp.name)
            tmp.seek(0)
            pdf_bytes = tmp.read()
        os.unlink(tmp.name)
        return pdf_bytes

    def save_temp_pdf(self):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        HTML(string=self.html_content).write_pdf(tmp.name)
        tmp.close()
        return tmp.name
