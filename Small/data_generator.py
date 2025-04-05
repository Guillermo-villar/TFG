#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def resolve_path(path):
    """
    Resolve a path that might be relative to the config file.
    
    Parameters:
    -----------
    path : str
        Path that might be relative
        
    Returns:
    --------
    resolved_path : str
        Absolute path
    """
    # If it's already an absolute path, return it
    if os.path.isabs(path):
        return path
    
    # Otherwise, make it relative to the current script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, path)
    
def ensure_dir_exists(file_path):
    """
    Ensure the directory for the specified file path exists.
    Creates the directory if it doesn't exist.
    
    Parameters:
    -----------
    file_path : str
        Path to a file
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def generate_synthetic_data(params, output_file):
    """
    Generate synthetic data for classification tasks and save it to a CSV file.
    
    Parameters:
    -----------
    params : dict
        Dictionary containing all parameters for data generation
        Required keys:
        - n_samples: Number of samples to generate
        - n_features: Total number of features
        - n_informative: Number of informative features
        - n_redundant: Number of redundant features
        - n_classes: Number of classes (for classification)
        - weights: List of class weights or None
        - random_state: Random seed for reproducibility
    output_file : str
        Name of the output CSV file
        
    Returns:
    --------
    data_info : dict
        Dictionary with information about the generated data and paths
    """
    # Resolve the output path
    resolved_output_file = resolve_path(output_file)
    
    # Ensure directory exists
    ensure_dir_exists(resolved_output_file)
    
    # Generate synthetic data
    x, y = make_classification(
        n_samples=params['n_samples'],
        n_features=params['n_features'],
        n_informative=params['n_informative'],
        n_redundant=params['n_redundant'],
        n_classes=params['n_classes'],
        weights=params.get('weights'),
        flip_y=params.get('flip_y', 0),
        random_state=params['random_state']
    )
    
    # Create feature column names
    feature_cols = [f'feature_{i+1}' for i in range(x.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(x, columns=feature_cols)
    df['target'] = y
    
    # Save to CSV
    df.to_csv(resolved_output_file, index=False)
    
    # Calculate data statistics
    class_stats = {
        "class_counts": np.bincount(y),
        "class_distribution": np.bincount(y) / len(y)
    }
    
    data_stats = {
        "samples": len(y),
        "features": x.shape[1],
        "classes": np.unique(y).size,
        "class_stats": class_stats,
        "data_shape": x.shape
    }
    
    data_info = {
        "file_path": resolved_output_file,
        "stats": data_stats,
        "params": params
    }
    
    return data_info

def print_data_stats(data_stats):
    """
    Print statistics about the generated data
    
    Parameters:
    -----------
    data_stats : dict
        Dictionary with statistics about the generated data
    """
    print(f"Data generation complete:")
    print(f"  + Total samples: {data_stats['samples']}")
    print(f"  + Features: {data_stats['features']}")
    print(f"  + Classes: {data_stats['classes']}")
    print(f"  + Data shape: {data_stats['data_shape']}")
    
    class_counts = data_stats["class_stats"]["class_counts"]
    class_distribution = data_stats["class_stats"]["class_distribution"]
    
    print("  + Class distribution:")
    for i, (count, percentage) in enumerate(zip(class_counts, class_distribution)):
        print(f"    - Class {i}: {count} samples ({percentage:.2%})")

if __name__ == "__main__":
    # Example parameters for generating synthetic data
    params = {
        'n_samples': 1000,
        'n_features': 20,
        'n_informative': 10,
        'n_redundant': 5,
        'n_classes': 2,
        'weights': None,  # Equal class weights
        'random_state': 42
    }
    
    output_file = 'datasets/synthetic_data.csv'
    
    # Generate data
    data_info = generate_synthetic_data(params, output_file)
    
    # Print statistics
    print_data_stats(data_info['stats'])
    print(f"Data saved to: {data_info['file_path']}")