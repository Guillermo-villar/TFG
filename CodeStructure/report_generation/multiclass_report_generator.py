#!/usr/bin/env python3
"""
Multiclass Reconstitution Report Generator

Generates specialized PDF reports for multiclass model reconstitution results,
focusing on ECOC decoding performance, statistical analysis, and model ensemble metrics.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import tempfile

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

class MulticlassReportGenerator:
    """Generates comprehensive PDF reports for multiclass reconstitution results"""
    
    def __init__(self, dataset_id, results):
        self.dataset_id = dataset_id
        self.results = results
        
        # Extract key information
        self.strategy = results.get('strategy', 'Unknown')
        self.n_classes = results.get('n_classes', 0)
        self.class_names = results.get('class_names', [])
        self.ecoc_matrix = np.array(results.get('ecoc_matrix', []))
        self.final_metrics = results.get('final_metrics', {})
        self.n_simulations = results.get('n_simulations', 0)
        self.test_size = results.get('test_size', 0.3)
        self.timestamp = results.get('timestamp', datetime.now().isoformat())
        
        # All simulation results for detailed analysis
        self.all_simulations = results.get('all_simulations', {})

    def generate_report(self):
        """Generate the complete multiclass reconstitution PDF report"""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch, cm
            from reportlab.lib.colors import HexColor, black, white, grey
            from reportlab.lib import colors
            from reportlab.graphics.shapes import Drawing, Rect
            from reportlab.graphics.charts.linecharts import HorizontalLineChart
            from reportlab.graphics.charts.barcharts import VerticalBarChart
            from reportlab.graphics import renderPDF
            
            # Create PDF in a permanent location (experiment directory)
            experiment_dir = f"/home/ubuntu/Downloads/Code/NeonGenesis/Small/datasets/IDs/{self.dataset_id}"
            if not os.path.exists(experiment_dir):
                # Fallback to results directory
                experiment_dir = "/home/ubuntu/Downloads/Code/results"
                os.makedirs(experiment_dir, exist_ok=True)
            
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"multiclass_reconstitution_report_{self.dataset_id}_{timestamp}.pdf"
            pdf_path = os.path.join(experiment_dir, pdf_filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(pdf_path, pagesize=A4, 
                                  rightMargin=72, leftMargin=72, 
                                  topMargin=72, bottomMargin=18)
            
            # Get styles
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                textColor=HexColor('#2E3440'),
                alignment=1  # Center
            )
            
            section_style = ParagraphStyle(
                'CustomSection',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=12,
                textColor=HexColor('#5E81AC'),
                borderWidth=1,
                borderColor=HexColor('#D8DEE9'),
                borderPadding=10,
                backColor=HexColor('#ECEFF4')
            )
            
            subsection_style = ParagraphStyle(
                'CustomSubsection',
                parent=styles['Heading3'],
                fontSize=14,
                spaceAfter=8,
                textColor=HexColor('#4C566A')
            )
            
            body_style = ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=6,
                textColor=HexColor('#2E3440')
            )
            
            # Build the document content
            story = []
            
            # Title page
            story.append(Paragraph("Multiclass Model Reconstitution Report", title_style))
            story.append(Spacer(1, 0.5*inch))
            
            # Dataset information
            story.append(Paragraph("Dataset Information", section_style))
            dataset_info = [
                ["Dataset ID:", self.dataset_id],
                ["Strategy:", self.strategy.upper()],
                ["Number of Classes:", str(self.n_classes)],
                ["Class Names:", ", ".join(self.class_names)],
                ["Test Size:", f"{self.test_size:.1%}"],
                ["Simulations:", str(self.n_simulations)],
                ["Generated:", datetime.fromisoformat(self.timestamp.replace('Z', '+00:00')).strftime("%Y-%m-%d %H:%M:%S")]
            ]
            
            dataset_table = Table(dataset_info, colWidths=[2*inch, 3*inch])
            dataset_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('BACKGROUND', (0, 0), (0, -1), HexColor('#F8F9FA')),
                ('PADDING', (0, 0), (-1, -1), 8)
            ]))
            story.append(dataset_table)
            story.append(Spacer(1, 0.3*inch))
            
            # ECOC Matrix
            story.append(Paragraph("Error-Correcting Output Codes (ECOC) Matrix", section_style))
            story.append(Paragraph(
                f"The {self.strategy.upper()} strategy decomposes the {self.n_classes}-class problem into "
                f"{self.ecoc_matrix.shape[1]} binary dichotomies using the following encoding matrix:",
                body_style
            ))
            story.append(Spacer(1, 0.1*inch))
            
            # Create ECOC matrix table
            ecoc_data = [["Class"] + [f"Dichotomy {i+1}" for i in range(self.ecoc_matrix.shape[1])]]
            for i, class_name in enumerate(self.class_names):
                row = [class_name] + [f"{self.ecoc_matrix[i, j]:+.0f}" for j in range(self.ecoc_matrix.shape[1])]
                ecoc_data.append(row)
            
            ecoc_table = Table(ecoc_data)
            ecoc_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#4C566A')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('BACKGROUND', (0, 1), (0, -1), HexColor('#ECEFF4')),
                ('PADDING', (0, 0), (-1, -1), 6)
            ]))
            story.append(ecoc_table)
            story.append(Spacer(1, 0.3*inch))
            
            # Performance Summary
            story.append(Paragraph("Reconstitution Performance Summary", section_style))
            
            # Main metrics table
            main_metrics = [
                ["Metric", "Average", "Std. Deviation", "95% Confidence Interval"],
                ["Accuracy", f"{self.final_metrics.get('avg_acc', 0):.4f}", 
                 f"±{self.final_metrics.get('std_acc', 0):.4f}",
                 f"[{self.final_metrics.get('avg_acc', 0) - 1.96*self.final_metrics.get('std_acc', 0):.4f}, "
                 f"{self.final_metrics.get('avg_acc', 0) + 1.96*self.final_metrics.get('std_acc', 0):.4f}]"],
                ["Balanced Accuracy", f"{self.final_metrics.get('avg_bal_acc', 0):.4f}", 
                 f"±{self.final_metrics.get('std_bal_acc', 0):.4f}",
                 f"[{self.final_metrics.get('avg_bal_acc', 0) - 1.96*self.final_metrics.get('std_bal_acc', 0):.4f}, "
                 f"{self.final_metrics.get('avg_bal_acc', 0) + 1.96*self.final_metrics.get('std_bal_acc', 0):.4f}]"],
                ["Cohen's Kappa", f"{self.final_metrics.get('avg_kappa', 0):.4f}", 
                 f"±{self.final_metrics.get('std_kappa', 0):.4f}",
                 f"[{self.final_metrics.get('avg_kappa', 0) - 1.96*self.final_metrics.get('std_kappa', 0):.4f}, "
                 f"{self.final_metrics.get('avg_kappa', 0) + 1.96*self.final_metrics.get('std_kappa', 0):.4f}]"],
                ["Geometric Mean", f"{self.final_metrics.get('avg_geom_mean', 0):.4f}", 
                 f"±{self.final_metrics.get('std_geom_mean', 0):.4f}",
                 f"[{self.final_metrics.get('avg_geom_mean', 0) - 1.96*self.final_metrics.get('std_geom_mean', 0):.4f}, "
                 f"{self.final_metrics.get('avg_geom_mean', 0) + 1.96*self.final_metrics.get('std_geom_mean', 0):.4f}]"],
                ["Sensitivity", f"{self.final_metrics.get('avg_sensitivity', 0):.4f}", 
                 f"±{self.final_metrics.get('std_sensitivity', 0):.4f}",
                 f"[{self.final_metrics.get('avg_sensitivity', 0) - 1.96*self.final_metrics.get('std_sensitivity', 0):.4f}, "
                 f"{self.final_metrics.get('avg_sensitivity', 0) + 1.96*self.final_metrics.get('std_sensitivity', 0):.4f}]"]
            ]
            
            metrics_table = Table(main_metrics, colWidths=[2*inch, 1.2*inch, 1.2*inch, 1.8*inch])
            metrics_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#5E81AC')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('PADDING', (0, 0), (-1, -1), 6),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold')
            ]))
            story.append(metrics_table)
            story.append(Spacer(1, 0.3*inch))
            
            # Performance interpretation
            story.append(Paragraph("Performance Interpretation", subsection_style))
            
            avg_acc = self.final_metrics.get('avg_acc', 0)
            avg_kappa = self.final_metrics.get('avg_kappa', 0)
            std_acc = self.final_metrics.get('std_acc', 0)
            
            # Interpret performance
            if avg_acc >= 0.9:
                performance_level = "Excellent"
                performance_color = "#5E81AC"
            elif avg_acc >= 0.8:
                performance_level = "Good"
                performance_color = "#88C0D0"
            elif avg_acc >= 0.7:
                performance_level = "Moderate"
                performance_color = "#EBCB8B"
            else:
                performance_level = "Poor"
                performance_color = "#BF616A"
            
            interpretation_text = f"""
            <b>Overall Performance:</b> {performance_level} ({avg_acc:.1%} accuracy)<br/>
            <b>Model Reliability:</b> {"High" if std_acc < 0.05 else "Moderate" if std_acc < 0.1 else "Variable"} 
            (±{std_acc:.1%} variation)<br/>
            <b>Agreement Level:</b> {"Excellent" if avg_kappa > 0.8 else "Good" if avg_kappa > 0.6 else "Moderate"} 
            (κ = {avg_kappa:.3f})<br/>
            <b>ECOC Effectiveness:</b> The {self.strategy.upper()} strategy successfully decomposed the {self.n_classes}-class problem 
            into {self.ecoc_matrix.shape[1]} binary problems with effective error correction.
            """
            
            story.append(Paragraph(interpretation_text, body_style))
            story.append(Spacer(1, 0.3*inch))
            
            # Detailed simulation results (if available)
            if self.all_simulations:
                story.append(Paragraph("Detailed Simulation Results", subsection_style))
                story.append(Spacer(1, 0.2*inch))
                
                # Create simulation results table
                sim_data = [["Simulation", "Accuracy", "Balanced Acc.", "Cohen's Kappa", "Geometric Mean", "Sensitivity"]]
                
                accuracy_sims = self.all_simulations.get('accuracy', [])
                bal_acc_sims = self.all_simulations.get('balanced_accuracy', [])
                kappa_sims = self.all_simulations.get('cohen_kappa', [])
                geom_mean_sims = self.all_simulations.get('geometric_mean', [])
                sensitivity_sims = self.all_simulations.get('sensitivity', [])
                
                for i in range(min(len(accuracy_sims), self.n_simulations)):
                    sim_data.append([
                        f"{i+1}",
                        f"{accuracy_sims[i]:.4f}" if i < len(accuracy_sims) else "N/A",
                        f"{bal_acc_sims[i]:.4f}" if i < len(bal_acc_sims) else "N/A",
                        f"{kappa_sims[i]:.4f}" if i < len(kappa_sims) else "N/A",
                        f"{geom_mean_sims[i]:.4f}" if i < len(geom_mean_sims) else "N/A",
                        f"{sensitivity_sims[i]:.4f}" if i < len(sensitivity_sims) else "N/A"
                    ])
                
                sim_table = Table(sim_data, colWidths=[1*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
                sim_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#4C566A')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('PADDING', (0, 0), (-1, -1), 4)
                ]))
                story.append(sim_table)
                story.append(Spacer(1, 0.3*inch))
            
            # Methodology section - ALWAYS execute, regardless of simulation data
            story.append(PageBreak())
            story.append(Paragraph("Detailed Model Reconstitution Process", section_style))
            
            explanation = [
                ("1. Loading Experiment Configuration", 
                 "The original multiclass experiment configuration is loaded. This includes the ECOC strategy (e.g., OVA, OVO), the number and names of classes, and the exact <b>ECOC coding matrix</b> that was used. This matrix is fundamental, as it defines how each binary classifier (dichotomy) contributes to the final decision."),
                
                ("2. Loading Original Dataset",
                 "The original CSV file containing the complete dataset with all classes is located and loaded. This serves as the starting point for simulating the training and validation process."),
                
                ("3. Loading Dichotomy Configurations",
                 "For each dichotomy (each column of the ECOC matrix), its optimal configuration is loaded from the hyperparameter optimization results. This file contains the model parameters (e.g., a Label Switching ensemble) that achieved the best performance for that specific binary classification task."),
                
                ("4. Simulation and Prediction",
                 "Multiple simulations are performed to ensure statistical robustness of the results. In each simulation:"),
                
                ("4a. Data Splitting and Standardization",
                 f"The original dataset is split into training and test sets (using {self.test_size:.0%} for testing). The data is standardized to have mean 0 and standard deviation 1, a standard practice for many Machine Learning models."),
                
                ("4b. Dichotomy Prediction",
                 "For each dichotomy, a model is instantiated with its optimal configuration and trained on the training data. Then, this model predicts the labels (-1 or 1) for the test set. The result is a matrix where each row is a test sample and each column is the prediction of a dichotomy."),
                
                ("5. Dichotomy Fusion (ECOC Decoding)",
                 "This is where the predictions from all binary classifiers are combined. For each test set sample:"),
                
                ("5a. Comparison with ECOC Matrix",
                 "The sample's prediction vector (e.g., <code>[1, -1, 1, ...]</code>) is compared with each of the 'codewords' of the ECOC matrix. Each 'codeword' (a row of the ECOC matrix) represents one of the original multiclass classes."),
                
                ("5b. Final Class Assignment",
                 "The 'distance' (usually Hamming distance, which counts how many predictions differ) is calculated between the sample's prediction vector and the 'codeword' of each class. The final class assigned to the sample is the one whose 'codeword' has the minimum distance. In other words, the class that is 'most compatible' with the set of dichotomy predictions is chosen."),
                
                ("6. Final Metrics Calculation",
                 "Once all test samples have a final class assigned, they are compared with the true labels to calculate performance metrics (Accuracy, Kappa, etc.). This process is repeated for all simulations, and the final results are averaged to obtain a reliable estimate of the reconstituted multiclass model's performance.")
            ]

            for title, text in explanation:
                if title.startswith("4a") or title.startswith("4b") or title.startswith("5a") or title.startswith("5b"):
                    story.append(Paragraph(f"<b>{title}</b>", ParagraphStyle(name='BodyIndent', parent=subsection_style, leftIndent=20)))
                    story.append(Paragraph(text, ParagraphStyle(name='BodyIndent', parent=body_style, leftIndent=20)))
                else:
                    story.append(Paragraph(f"<b>{title}</b>", subsection_style))
                    story.append(Paragraph(text, body_style))
                story.append(Spacer(1, 6))
            
            story.append(Spacer(1, 0.3*inch))
            
            # Footer information
            story.append(Spacer(1, 0.5*inch))
            footer_style = ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.grey,
                alignment=1
            )
            story.append(Paragraph(
                f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • "
                f"Multiclass Reconstitution Report • Dataset: {self.dataset_id}",
                footer_style
            ))
            
            # Build PDF
            doc.build(story)
            
            print(f"Multiclass reconstitution report generated: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            print(f"Error generating multiclass report: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
