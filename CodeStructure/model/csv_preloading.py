#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV Preprocessing Module

This module provides functions to preprocess CSV files from any format
into the standardized format expected by the ML system.

Expected format:
- Feature columns: feature_1, feature_2, ..., feature_n
- Target column: 'target' (binary: 0 or 1)
- No index column
- Numeric features only
"""

import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import logging

logger = logging.getLogger(__name__)

class CSVPreprocessor:
    """Class to handle CSV preprocessing with GUI interaction when needed"""
    
    def __init__(self, gui_enabled=True):
        self.gui_enabled = gui_enabled
        self.preprocessing_log = []
        
    def analyze_csv(self, file_path):
        """
        Analyze a CSV file to understand its structure and contents
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file to analyze
            
        Returns:
        --------
        dict : Analysis results including column info, data types, missing values, etc.
        """
        try:
            # Try different encoding options
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not read CSV with any common encoding")
            
            analysis = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'unique_counts': {col: df[col].nunique() for col in df.columns},
                'sample_data': df.head().to_dict(),
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
                'text_columns': [],  # Add missing text_columns field
                'binary_columns': [],
                'potential_targets': [],
                'encoding_used': encoding
            }
            
            # Identify binary columns
            for col in df.columns:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) == 2:
                    analysis['binary_columns'].append({
                        'column': col,
                        'values': unique_vals.tolist()
                    })
            
            # Identify text columns (high-cardinality string columns)
            for col in analysis['categorical_columns']:
                unique_count = analysis['unique_counts'][col]
                total_rows = df.shape[0]
                # Consider it a text column if it has high cardinality (>50% unique values)
                if unique_count > total_rows * 0.5 and unique_count > 10:
                    analysis['text_columns'].append({
                        'column': col,
                        'unique_count': unique_count,
                        'sample_values': df[col].dropna().head(3).tolist()
                    })
                    
            # Identify potential target columns
            for col in df.columns:
                unique_vals = df[col].dropna().unique()
                # Binary columns or columns with few unique values might be targets
                if len(unique_vals) <= 10:
                    analysis['potential_targets'].append({
                        'column': col,
                        'unique_count': len(unique_vals),
                        'values': unique_vals.tolist()[:10]  # Show max 10 values
                    })
                    
            self.preprocessing_log.append(f"Analyzed CSV: {df.shape[0]} rows, {df.shape[1]} columns")
            return analysis
            
        except Exception as e:
            error_msg = f"Error analyzing CSV file: {str(e)}"
            logger.error(error_msg)
            self.preprocessing_log.append(error_msg)
            raise
    
    def detect_multiclass(self, file_path):
        """
        Detects potential multiclass target columns in a CSV file.

        This function analyzes all columns to identify candidates for multiclass
        classification based on the number of unique values and additional heuristics.
        Also detects one-hot encoded targets (multiple binary columns where only one is 1 per row).
        
        IMPORTANT: If a clear binary target column already exists (named 'target', 'label', etc.),
        this function will return empty list to avoid false multiclass detection.

        Parameters:
        -----------
        file_path : str
            Path to the CSV file to analyze.

        Returns:
        --------
        list : A list of dictionaries, where each dictionary represents a
               potential multiclass column. Returns an empty list if no
               candidates are found or if a clear binary target already exists.
        """
        multiclass_candidates = []
        try:
            # Reuse the robust reading logic from analyze_csv
            df = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not read CSV with any common encoding")

            # FIRST: Detect one-hot encoded targets at the end of the CSV
            # Look for groups of binary columns (containing only 0s and 1s) where only one column is 1 per row
            n_cols = len(df.columns)
            one_hot_detected = False
            
            # Look for one-hot patterns in the last 2-20 columns
            for start_idx in range(max(0, n_cols - 20), n_cols - 1):
                for end_idx in range(start_idx + 2, n_cols + 1):
                    potential_columns = df.columns[start_idx:end_idx]
                    
                    if len(potential_columns) < 3:  # Need at least 3 classes for multiclass
                        continue
                    
                    # Check if all columns are binary (only 0s and 1s)
                    all_binary = True
                    for col in potential_columns:
                        unique_vals = df[col].dropna().unique()
                        # Must contain only 0 and 1 (and possibly NaN)
                        if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                            all_binary = False
                            break
                        # Must have both 0 and 1 (not just one value)
                        if len(unique_vals) < 2:
                            all_binary = False
                            break
                    
                    if not all_binary:
                        continue
                    
                    # Check if it follows one-hot pattern: exactly one 1 per row
                    subset_df = df[potential_columns].fillna(0)  # Fill NaN with 0
                    row_sums = subset_df.sum(axis=1)
                    
                    # Calculate how many rows have exactly one 1
                    one_hot_rows = (row_sums == 1).sum()
                    total_rows = len(subset_df)
                    
                    # Require at least 70% of rows to follow the one-hot pattern
                    one_hot_ratio = one_hot_rows / total_rows if total_rows > 0 else 0
                    
                    if one_hot_ratio >= 0.7:
                        # Additional checks for stronger confidence
                        
                        # Check 1: Column names suggest they're targets/classes
                        target_like_names = 0
                        class_like_patterns = ['class', 'target', 'label', 'category', 'type', 
                                             'group', 'cat_', 'is_', '_class', '_target']
                        for col in potential_columns:
                            col_lower = col.lower()
                            if any(pattern in col_lower for pattern in class_like_patterns):
                                target_like_names += 1
                        
                        # Check 2: Columns are at the very end of the dataset
                        is_at_end = (end_idx == n_cols)
                        
                        # Check 3: Each column has a reasonable distribution (not too sparse)
                        min_activation_ratio = min(df[col].sum() / len(df) for col in potential_columns)
                        
                        # Decision: Accept if it's at the end OR has target-like names OR good distribution
                        confidence_score = 0
                        if is_at_end:
                            confidence_score += 3
                        if target_like_names >= len(potential_columns) // 2:  # At least half have target-like names
                            confidence_score += 2
                        if min_activation_ratio >= 0.05:  # Each class appears in at least 5% of rows
                            confidence_score += 1
                        if one_hot_ratio >= 0.9:  # Very strong one-hot pattern
                            confidence_score += 2
                        
                        if confidence_score >= 3:
                            # Create class names from column names
                            class_names = [col.replace('_', ' ').replace('target', '').replace('class', '').strip() 
                                         for col in potential_columns]
                            # If names are empty or too similar, use the column names directly
                            if len(set(class_names)) < len(class_names) or any(not name for name in class_names):
                                class_names = list(potential_columns)
                            
                            self.preprocessing_log.append(
                                f"One-hot encoded target detected: {list(potential_columns)}"
                            )
                            self.preprocessing_log.append(
                                f"One-hot pattern confidence: {one_hot_ratio:.2f} ({one_hot_rows}/{total_rows} rows)"
                            )
                            
                            one_hot_target = {
                                'column': f"one_hot_target_{start_idx}_{end_idx}",  # Synthetic name
                                'n_classes': len(potential_columns),
                                'class_names': class_names,
                                'one_hot_columns': list(potential_columns),
                                'is_numeric': True,
                                'is_named_like_target': target_like_names > 0,
                                'column_position': start_idx,
                                'is_last_column': is_at_end,
                                'one_hot_ratio': one_hot_ratio,
                                'confidence_score': confidence_score,
                                'is_one_hot_encoded': True
                            }
                            multiclass_candidates.append(one_hot_target)
                            one_hot_detected = True
                            break
                
                if one_hot_detected:
                    break

            # If one-hot encoded target was found, return it (highest priority)
            if one_hot_detected:
                self.preprocessing_log.append(
                    f"Detected one-hot encoded multiclass target with {multiclass_candidates[0]['n_classes']} classes"
                )
                return multiclass_candidates

            # SECOND: Check if there's already a clear binary target column
            clear_target_keywords = ['target', 'label', 'class', 'outcome', 'result', 
                                   'prediction', 'output', 'y', 'diagnosis', 'response']
            
            for col in df.columns:
                col_lower = col.lower()
                unique_values = df[col].dropna().unique()
                n_unique = len(unique_values)
                
                # If we find a binary column with a target-like name, skip multiclass detection
                if (n_unique == 2 and 
                    any(keyword in col_lower for keyword in clear_target_keywords)):
                    self.preprocessing_log.append(
                        f"Found clear binary target column '{col}' - skipping multiclass detection"
                    )
                    return []  # Return empty list - this is clearly a binary classification problem

            # THIRD: Look for potential multiclass candidates, but be very strict
            min_classes = 3
            max_classes = 20  # Reduced from 50 to be more conservative

            for col in df.columns:
                unique_values = df[col].dropna().unique()
                n_classes = len(unique_values)

                if min_classes <= n_classes <= max_classes:
                    # Enhanced checks to avoid flagging feature columns as targets
                    is_likely_target = False  # Start with False, must prove it's a target
                    
                    # Check 1: Skip continuous float columns
                    if pd.api.types.is_float_dtype(df[col].dtype):
                        # If more than a couple of values are not whole numbers, treat as continuous
                        if (df[col].dropna() % 1 != 0).sum() > 2:
                            continue  # Skip this column
                    
                    # Check 2: Skip numeric columns that look like counts/measurements
                    if pd.api.types.is_numeric_dtype(df[col].dtype):
                        numeric_values = df[col].dropna()
                        if len(numeric_values) > 0:
                            # Check if it looks like a count/measurement feature
                            is_count_like = (
                                # All values are integers
                                all(val == int(val) for val in numeric_values) and
                                # Values are in a sequential pattern (like 0,1,2,3,4,5)
                                (max(numeric_values) - min(numeric_values)) < n_classes * 2 and
                                # Column name suggests it's a feature
                                any(keyword in col.lower() for keyword in [
                                    'age', 'count', 'number', 'num', 'pregnancy', 'pregnancies',
                                    'children', 'kids', 'experience', 'years', 'months', 'days',
                                    'level', 'score', 'rating', 'rank', 'size', 'length', 'height',
                                    'weight', 'amount', 'quantity', 'freq', 'frequency'
                                ])
                            )
                            
                            if is_count_like:
                                self.preprocessing_log.append(
                                    f"Skipping '{col}' - appears to be a count/measurement feature"
                                )
                                continue  # Skip this column
                    
                    # Check 3: Look for strong target column naming patterns
                    strong_target_keywords = ['target', 'label', 'class', 'species', 'category', 'type', 'grade', 
                                            'quality', 'status', 'condition', 'disease', 'cancer',
                                            'tumor', 'sentiment', 'emotion', 'classification']
                    is_strongly_named_target = any(keyword in col.lower() for keyword in strong_target_keywords)
                    
                    # Check 4: If it's in the last column, more likely to be a target
                    col_position = list(df.columns).index(col)
                    is_last_column = col_position == len(df.columns) - 1
                    
                    # DECISION LOGIC - Much more conservative:
                    if not pd.api.types.is_numeric_dtype(df[col].dtype):
                        # Non-numeric columns (strings) are good candidates if reasonably named
                        if is_strongly_named_target or is_last_column:
                            is_likely_target = True
                    elif is_strongly_named_target:
                        # Numeric columns only if very strongly named like a target
                        is_likely_target = True

                    if is_likely_target:
                        candidate = {
                            'column': col,
                            'n_classes': n_classes,
                            'class_names': sorted(list(unique_values)),
                            'is_numeric': pd.api.types.is_numeric_dtype(df[col].dtype),
                            'is_named_like_target': is_strongly_named_target,
                            'column_position': col_position,
                            'is_last_column': is_last_column,
                            'is_one_hot_encoded': False
                        }
                        multiclass_candidates.append(candidate)
                        self.preprocessing_log.append(
                            f"Found multiclass candidate '{col}' with {n_classes} classes."
                        )

            # FINAL CHECK: If no strong candidates found, don't flag as multiclass
            if not multiclass_candidates:
                self.preprocessing_log.append("No clear multiclass target candidates found")
            
            return multiclass_candidates

        except Exception as e:
            error_msg = f"Error during multiclass detection: {str(e)}"
            logger.error(error_msg)
            self.preprocessing_log.append(error_msg)
            return [] # Return empty list on error

    def identify_target_column(self, analysis):
        """
        Identify or ask user to specify the target column
        
        Parameters:
        -----------
        analysis : dict
            Results from analyze_csv()
            
        Returns:
        --------
        str : Name of the target column
        """
        # First, try automatic detection
        potential_targets = analysis['potential_targets']
        binary_columns = [col['column'] for col in analysis['binary_columns']]
        
        # Prioritize binary columns
        if binary_columns:
            # Check common target column names in binary columns
            common_names = ['target', 'label', 'class', 'y', 'outcome', 'result']
            for name in common_names:
                if name.lower() in [col.lower() for col in binary_columns]:
                    matching_col = next(col for col in binary_columns if col.lower() == name.lower())
                    self.preprocessing_log.append(f"Auto-detected target column: {matching_col}")
                    return matching_col
            
            # If only one binary column, use it
            if len(binary_columns) == 1:
                self.preprocessing_log.append(f"Auto-detected target column (only binary): {binary_columns[0]}")
                return binary_columns[0]
        
        # If GUI is enabled, ask user
        if self.gui_enabled:
            return self._ask_user_target_column(analysis)
        else:
            # Default to last column if no GUI
            target_col = analysis['columns'][-1]
            self.preprocessing_log.append(f"Using last column as target: {target_col}")
            return target_col
    
    def _ask_user_target_column(self, analysis):
        """
        Show GUI dialog to let user select target column
        """
        root = tk.Tk()
        root.withdraw()  # Hide main window
        
        # Create selection dialog
        dialog = tk.Toplevel(root)
        dialog.title("Select Target Column")
        dialog.geometry("600x400")
        dialog.transient(root)
        dialog.grab_set()
        
        selected_column = tk.StringVar()
        
        # Instructions
        instructions = tk.Label(dialog, 
                               text="Please select the target column for your machine learning model:",
                               font=('Arial', 12, 'bold'))
        instructions.pack(pady=10)
        
        # Create frame for column selection
        frame = ttk.Frame(dialog)
        frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create treeview to show column information
        columns = ['Column Name', 'Type', 'Unique Values', 'Sample Values']
        tree = ttk.Treeview(frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=140)
        
        # Populate treeview with column information
        for col_name in analysis['columns']:
            dtype = str(analysis['dtypes'][col_name])
            unique_count = analysis['unique_counts'][col_name]
            
            # Get sample values
            sample_values = []
            for target_info in analysis['potential_targets']:
                if target_info['column'] == col_name:
                    sample_values = target_info['values'][:3]  # First 3 values
                    break
            
            if not sample_values and col_name in analysis['sample_data']:
                sample_values = list(analysis['sample_data'][col_name].values())[:3]
            
            sample_str = ', '.join([str(v) for v in sample_values])
            if len(sample_str) > 30:
                sample_str = sample_str[:27] + "..."
            
            tree.insert('', 'end', values=(col_name, dtype, unique_count, sample_str))
        
        tree.pack(fill='both', expand=True)
        
        # Selection handling
        def on_select(event):
            selection = tree.selection()
            if selection:
                item = tree.item(selection[0])
                selected_column.set(item['values'][0])
        
        tree.bind('<<TreeviewSelect>>', on_select)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        result = {'column': None}
        
        def confirm():
            if selected_column.get():
                result['column'] = selected_column.get()
                dialog.destroy()
                root.quit()
            else:
                messagebox.showwarning("Warning", "Please select a column first")
        
        def cancel():
            dialog.destroy()
            root.quit()
        
        ttk.Button(button_frame, text="Confirm", command=confirm).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side='left', padx=5)
        
        # Highlight binary columns
        for i, col_name in enumerate(analysis['columns']):
            if col_name in [col['column'] for col in analysis['binary_columns']]:
                tree.set(tree.get_children()[i], 'Column Name', f"🎯 {col_name}")
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        root.mainloop()
        root.destroy()
        
        if result['column']:
            self.preprocessing_log.append(f"User selected target column: {result['column']}")
            return result['column']
        else:
            raise ValueError("No target column selected")
    
    def preprocess_target_column(self, df, target_column, analysis):
        """
        Convert target column to binary format (0, 1)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe to process
        target_column : str
            Name of the target column
        analysis : dict
            Analysis results
            
        Returns:
        --------
        pandas.Series : Processed target column (0, 1)
        dict : Mapping information for the conversion
        """
        target_series = df[target_column].copy()
        unique_values = target_series.dropna().unique()
        
        # If already binary (0, 1), return as is
        if set(unique_values) == {0, 1} or set(unique_values) == {0.0, 1.0}:
            self.preprocessing_log.append("Target column already in binary format (0, 1)")
            return target_series.astype(int), {'mapping': {0: 0, 1: 1}}
        
        # If binary but different values
        if len(unique_values) == 2:
            values = sorted(unique_values)
            mapping = {values[0]: 0, values[1]: 1}
            target_series = target_series.map(mapping)
            self.preprocessing_log.append(f"Mapped target values: {mapping}")
            return target_series, {'mapping': mapping}
        
        # If more than 2 values, ask user or use heuristics
        if len(unique_values) > 2:
            if self.gui_enabled:
                return self._ask_user_target_mapping(target_series, unique_values)
            else:
                # Default: treat as classification problem, map most frequent to 0
                value_counts = target_series.value_counts()
                majority_class = value_counts.index[0]
                mapping = {val: (0 if val == majority_class else 1) for val in unique_values}
                target_series = target_series.map(mapping)
                self.preprocessing_log.append(f"Auto-mapped multi-class to binary: {mapping}")
                return target_series, {'mapping': mapping}
        
        raise ValueError(f"Cannot process target column with values: {unique_values}")
    
    def _ask_user_target_mapping(self, target_series, unique_values):
        """
        Ask user how to map target values to binary
        """
        root = tk.Tk()
        root.withdraw()
        
        dialog = tk.Toplevel(root)
        dialog.title("Target Column Mapping")
        dialog.geometry("500x400")
        dialog.transient(root)
        dialog.grab_set()
        
        tk.Label(dialog, text=f"Target column has {len(unique_values)} unique values.", 
                font=('Arial', 12, 'bold')).pack(pady=10)
        tk.Label(dialog, text="Please map them to binary values (0 or 1):").pack(pady=5)
        
        # Show value counts
        value_counts = target_series.value_counts()
        
        frame = ttk.Frame(dialog)
        frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create mapping interface
        mapping_vars = {}
        for i, value in enumerate(unique_values):
            row_frame = ttk.Frame(frame)
            row_frame.pack(fill='x', pady=5)
            
            count = value_counts.get(value, 0)
            ttk.Label(row_frame, text=f"'{value}' (appears {count} times):").pack(side='left')
            
            var = tk.StringVar(value="0" if i == 0 else "1")  # Default mapping
            mapping_vars[value] = var
            
            ttk.Radiobutton(row_frame, text="0", variable=var, value="0").pack(side='right', padx=5)
            ttk.Radiobutton(row_frame, text="1", variable=var, value="1").pack(side='right', padx=5)
        
        result = {'mapping': None}
        
        def confirm():
            mapping = {}
            for value, var in mapping_vars.items():
                mapping[value] = int(var.get())
            
            # Check if mapping is valid (has both 0 and 1)
            mapped_values = set(mapping.values())
            if len(mapped_values) != 2 or mapped_values != {0, 1}:
                messagebox.showwarning("Warning", "Please map values to both 0 and 1")
                return
            
            result['mapping'] = mapping
            dialog.destroy()
            root.quit()
        
        ttk.Button(dialog, text="Confirm Mapping", command=confirm).pack(pady=20)
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        root.mainloop()
        root.destroy()
        
        if result['mapping']:
            mapped_series = target_series.map(result['mapping'])
            self.preprocessing_log.append(f"User mapped target values: {result['mapping']}")
            return mapped_series, {'mapping': result['mapping']}
        else:
            raise ValueError("No mapping provided")
    
    def preprocess_feature_columns(self, df, target_column, analysis):
        """
        Process feature columns to ensure they're numeric and properly formatted
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe to process
        target_column : str
            Name of the target column (to exclude)
        analysis : dict
            Analysis results
            
        Returns:
        --------
        pandas.DataFrame : Processed feature columns
        """
        # Get feature columns (all except target)
        feature_columns = [col for col in df.columns if col != target_column]
        feature_df = df[feature_columns].copy()
        
        processing_log = []
        
        for col in feature_columns:
            if col in analysis['numeric_columns']:
                # Handle missing values in numeric columns
                if feature_df[col].isnull().sum() > 0:
                    # Fill with median
                    median_val = feature_df[col].median()
                    feature_df[col].fillna(median_val, inplace=True)
                    processing_log.append(f"Filled {col} missing values with median: {median_val}")
            
            elif col in analysis['categorical_columns']:
                # Convert categorical to numeric
                if feature_df[col].dtype == 'object':
                    unique_vals = feature_df[col].dropna().unique()
                    
                    # If binary categorical, convert to 0/1
                    if len(unique_vals) == 2:
                        mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                        feature_df[col] = feature_df[col].map(mapping)
                        processing_log.append(f"Binary encoded {col}: {mapping}")
                    
                    # If few categories, use label encoding
                    elif len(unique_vals) <= 10:
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        feature_df[col] = le.fit_transform(feature_df[col].fillna('missing'))
                        processing_log.append(f"Label encoded {col}: {len(unique_vals)} categories")
                    
                    # If many categories, ask user or drop
                    else:
                        if self.gui_enabled and self._ask_user_high_cardinality(col, len(unique_vals)):
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            feature_df[col] = le.fit_transform(feature_df[col].fillna('missing'))
                            processing_log.append(f"Label encoded high-cardinality {col}: {len(unique_vals)} categories")
                        else:
                            feature_df.drop(col, axis=1, inplace=True)
                            processing_log.append(f"Dropped high-cardinality column: {col}")
                            continue
                
                # Fill remaining missing values
                if feature_df[col].isnull().sum() > 0:
                    mode_val = feature_df[col].mode()[0] if len(feature_df[col].mode()) > 0 else 0
                    feature_df[col].fillna(mode_val, inplace=True)
                    processing_log.append(f"Filled {col} missing values with mode: {mode_val}")
        
        # Ensure all columns are numeric
        for col in feature_df.columns:
            if not pd.api.types.is_numeric_dtype(feature_df[col]):
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
                if feature_df[col].isnull().sum() > 0:
                    feature_df[col].fillna(0, inplace=True)
                    processing_log.append(f"Converted {col} to numeric, filled NaN with 0")
        
        # Rename columns to expected format
        feature_df.columns = [f'feature_{i+1}' for i in range(len(feature_df.columns))]
        processing_log.append(f"Renamed columns to feature_1 through feature_{len(feature_df.columns)}")
        
        self.preprocessing_log.extend(processing_log)
        return feature_df
    
    def _ask_user_high_cardinality(self, column_name, cardinality):
        """
        Ask user whether to keep high-cardinality categorical column
        """
        if not self.gui_enabled:
            return False
            
        response = messagebox.askyesno(
            "High Cardinality Column",
            f"Column '{column_name}' has {cardinality} unique values.\n\n"
            f"This might not be suitable for machine learning.\n\n"
            f"Do you want to keep this column?\n"
            f"(Yes = encode as numbers, No = remove column)"
        )
        return response
    
    def process_csv_file(self, input_path, output_path=None):
        """
        Complete preprocessing pipeline for a CSV file
        
        Parameters:
        -----------
        input_path : str
            Path to input CSV file
        output_path : str, optional
            Path for output CSV file (if None, will use input path with _processed suffix)
            
        Returns:
        --------
        dict : Processing results including output path, logs, and statistics
        """
        try:
            # Step 1: Analyze the CSV
            self.preprocessing_log = []
            self.preprocessing_log.append(f"Starting preprocessing for: {input_path}")
            
            analysis = self.analyze_csv(input_path)
            
            # Step 2: Load the data
            df = pd.read_csv(input_path, encoding=analysis['encoding_used'])
            
            # Step 3: Identify target column
            target_column = self.identify_target_column(analysis)
            
            # Step 4: Process target column
            processed_target, target_mapping = self.preprocess_target_column(df, target_column, analysis)
            
            # Step 5: Process feature columns
            processed_features = self.preprocess_feature_columns(df, target_column, analysis)
            
            # Step 6: Combine processed data
            final_df = processed_features.copy()
            final_df['target'] = processed_target
            
            # Step 7: Save processed data
            if output_path is None:
                base_name = os.path.splitext(input_path)[0]
                output_path = f"{base_name}_processed.csv"
            
            final_df.to_csv(output_path, index=False)
            
            # Final statistics
            final_stats = {
                'original_shape': analysis['shape'],
                'processed_shape': final_df.shape,
                'target_distribution': final_df['target'].value_counts().to_dict(),
                'features_count': len(processed_features.columns),
                'target_mapping': target_mapping
            }
            
            self.preprocessing_log.append(f"Preprocessing completed successfully")
            self.preprocessing_log.append(f"Original shape: {final_stats['original_shape']}")
            self.preprocessing_log.append(f"Processed shape: {final_stats['processed_shape']}")
            self.preprocessing_log.append(f"Target distribution: {final_stats['target_distribution']}")
            self.preprocessing_log.append(f"Output saved to: {output_path}")
            
            return {
                'success': True,
                'output_path': output_path,
                'statistics': final_stats,
                'processing_log': self.preprocessing_log.copy(),
                'analysis': analysis
            }
            
        except Exception as e:
            error_msg = f"Error during preprocessing: {str(e)}"
            logger.error(error_msg)
            self.preprocessing_log.append(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'processing_log': self.preprocessing_log.copy()
            }


def preprocess_csv_auto(input_path, output_path=None, gui_enabled=True):
    """
    Convenience function for automatic CSV preprocessing
    
    Parameters:
    -----------
    input_path : str
        Path to input CSV file
    output_path : str, optional
        Path for output CSV file
    gui_enabled : bool
        Whether to use GUI for user interaction
        
    Returns:
    --------
    dict : Processing results
    """
    processor = CSVPreprocessor(gui_enabled=gui_enabled)
    return processor.process_csv_file(input_path, output_path)


def validate_processed_csv(file_path):
    """
    Validate that a CSV file is in the expected format for the ML system
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file to validate
        
    Returns:
    --------
    dict : Validation results
    """
    try:
        df = pd.read_csv(file_path)
        
        issues = []
        warnings = []
        
        # Check if target column exists
        if 'target' not in df.columns:
            issues.append("Missing 'target' column")
        else:
            # Check target column format
            unique_targets = df['target'].unique()
            if not set(unique_targets).issubset({0, 1}):
                issues.append(f"Target column should contain only 0 and 1, found: {unique_targets}")
        
        # Check feature columns
        feature_cols = [col for col in df.columns if col != 'target']
        expected_feature_cols = [f'feature_{i+1}' for i in range(len(feature_cols))]
        
        if feature_cols != expected_feature_cols:
            warnings.append("Feature columns don't follow expected naming convention (feature_1, feature_2, ...)")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            issues.append(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
        
        # Check for non-numeric features
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"Non-numeric feature column: {col}")
        
        is_valid = len(issues) == 0;
        
        return {
            'valid': is_valid,
            'issues': issues,
            'warnings': warnings,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'target_distribution': df['target'].value_counts().to_dict() if 'target' in df.columns else None
        }
        
    except Exception as e:
        return {
            'valid': False,
            'issues': [f"Error reading file: {str(e)}"],
            'warnings': [],
            'shape': None,
            'columns': None,
            'target_distribution': None
        }


def create_multiclass_dichotomies(input_csv_path, multiclass_column_info, strategy='OVO', output_dir=None):
    """
    Create binary dichotomies from a multiclass dataset using ECOC decomposition.
    Uses the dataset ID system to store dichotomies in the correct location.
    
    Parameters:
    -----------
    input_csv_path : str
        Path to the input CSV file with multiclass target
    multiclass_column_info : dict
        Information about the multiclass column from detect_multiclass()
    strategy : str
        Decomposition strategy: 'OVO' (One-vs-One) or 'OVA' (One-vs-All)
    output_dir : str, optional
        Directory to save the dichotomy files. If None, uses dataset ID system
        
    Returns:
    --------
    dict : Information about the created dichotomies
    """
    import sys
    import os
    import numpy as np
    import pandas as pd
    
    try:
        # Import dataset ID functions
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_file_dir)
        from data_generator import generate_real_data_id, get_dataset_directory_structure
        
        # Generate dataset ID for this multiclass dataset
        dataset_id = generate_real_data_id(input_csv_path)
        
        # Get dataset directory structure
        dataset_paths = get_dataset_directory_structure(dataset_id)
        dataset_dir = dataset_paths["dataset_dir"]
        
        # Create dataset directory if it doesn't exist
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Check if dichotomies already exist for this strategy
        dichotomy_pattern = os.path.join(dataset_dir, f"{dataset_id}_{strategy.lower()}_dichotomy_*.csv")
        existing_files = []
        dichotomy_index = 1
        while True:
            dichotomy_file = os.path.join(dataset_dir, f"{dataset_id}_{strategy.lower()}_dichotomy_{dichotomy_index:02d}.csv")
            if os.path.exists(dichotomy_file):
                existing_files.append(dichotomy_file)
                dichotomy_index += 1
            else:
                break
        
        # If dichotomies already exist, return them
        if existing_files:
            # When loading from cache, we still need to provide n_classes, class_names, and code_matrix
            df_for_n_classes = pd.read_csv(input_csv_path)
            
            if multiclass_column_info.get('is_one_hot_encoded', False):
                n_classes = len(multiclass_column_info['class_names'])
                class_names = multiclass_column_info['class_names']
            else:
                target_column_for_n_classes = multiclass_column_info['column']
                n_classes = df_for_n_classes[target_column_for_n_classes].nunique()
                class_names = sorted(df_for_n_classes[target_column_for_n_classes].dropna().unique())

            # Re-create the ECOC encoder to get the code matrix
            try:
                from ecoc import ECOC
            except ImportError as e:
                raise ImportError(f"Could not import ECOC. Error: {e}")
            
            labels = np.arange(n_classes)
            strategy_upper = strategy.upper()
            ecoc = ECOC(encoding=strategy_upper, labels=labels)
            code_matrix = ecoc._code_matrix

            # Re-create the dichotomies information by reading the files
            dichotomies = []
            for i, f in enumerate(existing_files):
                binary_df = pd.read_csv(f)
                positive_classes = [class_names[j] for j in range(n_classes) if code_matrix[j, i] == 1]
                negative_classes = [class_names[j] for j in range(n_classes) if code_matrix[j, i] == -1]
                
                dichotomies.append({
                    'index': i + 1,
                    'file': f,
                    'positive_classes': positive_classes,
                    'negative_classes': negative_classes,
                    'n_instances': len(binary_df),
                    'n_positive': (binary_df['target'] == 1).sum(),
                    'n_negative': (binary_df['target'] == 0).sum()
                })

            return {
                'success': True,
                'strategy': strategy,
                'dataset_id': dataset_id,
                'n_classes': n_classes,
                'class_names': class_names,
                'n_dichotomies': len(existing_files),
                'dichotomies': dichotomies,
                'dichotomy_files': existing_files,
                'code_matrix': code_matrix.tolist(),
                'output_dir': dataset_dir,
                'base_name': dataset_id,
                'from_cache': True
            }

        # Read the input CSV
        df = pd.read_csv(input_csv_path)
        
        # Handle one-hot encoded targets
        if multiclass_column_info.get('is_one_hot_encoded', False):
            one_hot_columns = multiclass_column_info['one_hot_columns']
            class_names = multiclass_column_info['class_names']
            
            # Convert one-hot to single column
            target_values = []
            for idx, row in df.iterrows():
                # Find which class is active (has value 1)
                active_classes = []
                for i, col in enumerate(one_hot_columns):
                    if row[col] == 1:
                        active_classes.append(i)
                
                if len(active_classes) == 1:
                    target_values.append(active_classes[0])
                elif len(active_classes) == 0:
                    target_values.append(-1)  # No class active
                else:
                    target_values.append(active_classes[0])  # Take first if multiple
            
            # Remove one-hot columns and add single target column
            feature_df = df.drop(columns=one_hot_columns)
            feature_df['multiclass_target'] = target_values
            target_column = 'multiclass_target'
            
        else:
            # Regular multiclass column
            target_column = multiclass_column_info['column']
            feature_df = df.copy()
        
        # Get feature columns (everything except target)
        feature_columns = [col for col in feature_df.columns if col != target_column]
        
        # Get unique classes and create mapping
        unique_classes = sorted(feature_df[target_column].dropna().unique())
        n_classes = len(unique_classes)
        
        if n_classes < 3:
            raise ValueError(f"Not enough classes for multiclass decomposition: {n_classes}")
        
        # Create class mapping
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        
        # Import ECOC from MC_Label_Switching
        # Get the absolute path to the MC_Label_Switching libraries directory
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_file_dir))  # Go up to /home/ubuntu/Downloads/Code
        mc_path = os.path.join(project_root, 'MC_Label_Switching-main', 'libraries')
        
        if not os.path.exists(mc_path):
            raise ImportError(f"MC_Label_Switching libraries directory not found at: {mc_path}")
        
        if mc_path not in sys.path:
            sys.path.insert(0, mc_path)
        
        try:
            from ecoc import ECOC
        except ImportError as e:
            raise ImportError(f"Could not import ECOC from {mc_path}. Error: {e}")
        
        # Create ECOC encoder
        labels = np.arange(n_classes)
        # Convert strategy to uppercase as ECOC expects 'OVO' or 'OVA', not 'ovo' or 'ova'
        strategy_upper = strategy.upper()
        ecoc = ECOC(encoding=strategy_upper, labels=labels)
        code_matrix = ecoc._code_matrix
        
        dichotomies = []
        dichotomy_files = []
        
        # Create each dichotomy
        for dict_idx in range(code_matrix.shape[1]):
            dichotomy_column = code_matrix[:, dict_idx]
            
            # Create binary dataset for this dichotomy
            binary_data = []
            
            for idx, row in feature_df.iterrows():
                if pd.isna(row[target_column]):
                    continue
                    
                original_class = row[target_column]
                if original_class not in class_to_idx:
                    continue
                    
                class_idx = class_to_idx[original_class]
                dichotomy_value = dichotomy_column[class_idx]
                
                # Skip instances with 0 (excluded from this dichotomy)
                if dichotomy_value == 0:
                    continue
                
                # Convert to binary (1 -> 1, -1 -> 0)
                binary_target = 1 if dichotomy_value == 1 else 0
                
                # Create row with features + binary target
                binary_row = {}
                for feat_col in feature_columns:
                    binary_row[feat_col] = row[feat_col]
                binary_row['target'] = binary_target
                
                binary_data.append(binary_row)
            
            if len(binary_data) > 0:
                # Create DataFrame and save
                binary_df = pd.DataFrame(binary_data)
                
                # Create filename using dataset ID system
                dichotomy_file = os.path.join(dataset_dir, f"{dataset_id}_{strategy.lower()}_dichotomy_{dict_idx+1:02d}.csv")
                binary_df.to_csv(dichotomy_file, index=False)
                
                dichotomy_files.append(dichotomy_file)
                
                # Store dichotomy info
                positive_classes = [unique_classes[i] for i in range(n_classes) if dichotomy_column[i] == 1]
                negative_classes = [unique_classes[i] for i in range(n_classes) if dichotomy_column[i] == -1]
                
                dichotomies.append({
                    'index': dict_idx + 1,
                    'file': dichotomy_file,
                    'positive_classes': positive_classes,
                    'negative_classes': negative_classes,
                    'n_instances': len(binary_df),
                    'n_positive': (binary_df['target'] == 1).sum(),
                    'n_negative': (binary_df['target'] == 0).sum()
                })
        
        # Also save the original dataset in the ID directory for reference
        # Use the original filename instead of the hash-based name
        original_filename = os.path.basename(input_csv_path)
        original_dataset_file = os.path.join(dataset_dir, original_filename)
        if not os.path.exists(original_dataset_file):
            df.to_csv(original_dataset_file, index=False)
        
        return {
            'success': True,
            'strategy': strategy,
            'dataset_id': dataset_id,
            'n_classes': n_classes,
            'class_names': unique_classes,
            'n_dichotomies': len(dichotomies),
            'dichotomies': dichotomies,
            'dichotomy_files': dichotomy_files,
            'code_matrix': code_matrix.tolist(),
            'output_dir': dataset_dir,
            'base_name': dataset_id,
            'from_cache': False
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'traceback': str(e.__class__.__name__)
        }


if __name__ == "__main__":
    # Example usage
    print("CSV Preprocessing Module")
    print("This module helps convert any CSV format to the expected ML system format.")
    print("\nExpected format:")
    print("- Feature columns: feature_1, feature_2, ..., feature_n")
    print("- Target column: 'target' (binary: 0 or 1)")
    print("- No missing values")
    print("- All features are numeric")
