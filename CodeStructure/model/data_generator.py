#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import hashlib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def generate_dataset_id(params, data_type="synth"):
    """
    Generate a unique dataset ID based on dataset parameters.
    
    Parameters:
    -----------
    params : dict
        Dictionary containing dataset parameters
    data_type : str
        Type of dataset ("synth" or "real")
        
    Returns:
    --------
    dataset_id : str
        Unique identifier for the dataset
    """
    # Create a string representation of parameters for hashing
    if data_type == "synth":
        # For synthetic data, use the generation parameters
        param_str = f"{params.get('n_samples', 0)}_{params.get('n_features', 0)}_{params.get('n_informative', 0)}_{params.get('n_redundant', 0)}_{params.get('n_classes', 2)}_{params.get('random_state', 42)}"
        if params.get('weights'):
            param_str += f"_weights_{'_'.join(map(str, params['weights']))}"
    else:
        # For real data, use the file path
        param_str = str(params.get('file_path', 'unknown'))
    
    # Generate hash of parameters
    hash_obj = hashlib.md5(param_str.encode())
    dataset_hash = hash_obj.hexdigest()[:8]  # Use first 8 characters
    
    # Create dataset ID with type prefix
    dataset_id = f"{data_type}_{dataset_hash}"
    
    return dataset_id

def get_dataset_directory_structure(dataset_id, base_datasets_dir=None):
    """
    Get the directory structure for a dataset ID.
    
    Parameters:
    -----------
    dataset_id : str
        The dataset ID
    base_datasets_dir : str, optional
        Base directory for datasets
        
    Returns:
    --------
    dict : Dictionary with all relevant paths for the dataset
    """
    if base_datasets_dir is None:
        base_datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    
    dataset_dir = os.path.join(base_datasets_dir, "IDs", dataset_id)
    runs_dir = os.path.join(dataset_dir, "runs")
    
    return {
        "dataset_dir": dataset_dir,
        "runs_dir": runs_dir,
        "original_csv": os.path.join(dataset_dir, f"{dataset_id}_original.csv"),
        "processed_csv": os.path.join(dataset_dir, f"{dataset_id}_processed.csv"),
        "best_runner_json": os.path.join(dataset_dir, f"best_runner_{dataset_id}.json")
    }

def get_run_directory_structure(dataset_id, timestamp=None, base_datasets_dir=None):
    """
    Get the directory structure for a specific run within a dataset.
    
    Parameters:
    -----------
    dataset_id : str
        The dataset ID
    timestamp : str, optional
        Timestamp for the run (if None, generates current timestamp)
    base_datasets_dir : str, optional
        Base directory for datasets
        
    Returns:
    --------
    dict : Dictionary with all relevant paths for the run
    """
    import datetime
    
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    dataset_paths = get_dataset_directory_structure(dataset_id, base_datasets_dir)
    run_dir = os.path.join(dataset_paths["runs_dir"], timestamp)
    
    return {
        "dataset_id": dataset_id,
        "timestamp": timestamp,
        "run_dir": run_dir,
        "terminal_output": os.path.join(run_dir, "terminal_output.txt"),
        "results_csv": os.path.join(dataset_paths["dataset_dir"], "test_results.csv"),  # Dataset level, not run level
        "dataset_paths": dataset_paths
    }

def find_latest_run_across_all_datasets(base_datasets_dir=None):
    """
    Find the latest experiment run across all datasets.
    
    Parameters:
    -----------
    base_datasets_dir : str, optional
        Base directory for datasets
        
    Returns:
    --------
    dict : Information about the latest run or None if no runs found
    """
    import os
    
    if base_datasets_dir is None:
        base_datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    
    ids_dir = os.path.join(base_datasets_dir, "IDs")
    
    if not os.path.exists(ids_dir):
        return None
    
    latest_run = None
    latest_time = 0
    
    # Scan all dataset directories
    for dataset_id in os.listdir(ids_dir):
        dataset_dir = os.path.join(ids_dir, dataset_id)
        if not os.path.isdir(dataset_dir):
            continue
            
        runs_dir = os.path.join(dataset_dir, "runs")
        if not os.path.exists(runs_dir):
            continue
            
        # Check all run directories in this dataset
        for run_timestamp in os.listdir(runs_dir):
            run_dir = os.path.join(runs_dir, run_timestamp)
            if not os.path.isdir(run_dir):
                continue
                
            # Get modification time of the run directory
            try:
                mtime = os.path.getmtime(run_dir)
                if mtime > latest_time:
                    latest_time = mtime
                    latest_run = {
                        "dataset_id": dataset_id,
                        "timestamp": run_timestamp,
                        "run_dir": run_dir,
                        "terminal_output": os.path.join(run_dir, "terminal_output.txt"),
                        "results_csv": os.path.join(run_dir, "test_results.csv"),
                        "dataset_dir": dataset_dir
                    }
            except OSError:
                continue
    
    return latest_run

def setup_real_dataset_structure(file_path, base_datasets_dir=None):
    """
    Set up directory structure for a real dataset file.
    
    Parameters:
    -----------
    file_path : str
        Path to the real dataset file
    base_datasets_dir : str, optional
        Base directory for datasets
        
    Returns:
    --------
    data_info : dict
        Dictionary with information about the dataset and paths
    """
    import shutil
    
    # Generate dataset ID for real data from the original file
    dataset_id = generate_real_data_id(file_path)
    
    # Get directory structure
    paths = get_dataset_directory_structure(dataset_id, base_datasets_dir)
    
    # Ensure the dataset-specific directory exists
    ensure_dir_exists(os.path.join(paths["dataset_dir"], "file.tmp"))
    
    # Define paths for original and processed datasets
    original_dst_path = paths["original_csv"]
    processed_dst_path = paths["processed_csv"] 

    # Copy the original file for reference, if it doesn't exist
    if not os.path.exists(original_dst_path):
        shutil.copy2(file_path, original_dst_path)

    # Copy the original file to be the dataset file for processing, if it doesn't exist
    if not os.path.exists(processed_dst_path):
        shutil.copy2(file_path, processed_dst_path)

    data_info = {
        "dataset_id": dataset_id,
        "dataset_dir": paths["dataset_dir"],
        "file_path": processed_dst_path, # This points to {dataset_id}_processed.csv
        "original_path": file_path,
        "original_saved_path": original_dst_path,
        "paths": paths
    }
    
    return data_info

def generate_real_data_id(file_path):
    """
    Generate a unique dataset ID for real data files based on intrinsic dataset characteristics.
    
    Parameters:
    -----------
    file_path : str
        Path to the real data file
        
    Returns:
    --------
    dataset_id : str
        Unique identifier for the dataset based on its content and structure
    """
    import pandas as pd
    import numpy as np
    
    try:
        df = pd.read_csv(file_path)
        
        # 1. Metadatos básicos del dataset
        metadata = {
            'shape': df.shape,
            'columns': sorted(df.columns.tolist()),
            'dtypes': {col: str(dtype) for col, dtype in sorted(df.dtypes.items())}
        }
        
        # 2. Estadísticas rápidas y determinísticas
        stats = {}
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                # Para columnas numéricas: media, std, min, max
                stats[col] = {
                    'mean': round(df[col].mean(), 6) if not df[col].isna().all() else None,
                    'std': round(df[col].std(), 6) if not df[col].isna().all() else None,
                    'min': df[col].min() if not df[col].isna().all() else None,
                    'max': df[col].max() if not df[col].isna().all() else None
                }
            else:
                # Para columnas categóricas: número de únicos y muestra de valores
                unique_vals = df[col].dropna().unique()
                stats[col] = {
                    'unique_count': len(unique_vals),
                    'sample_values': sorted([str(v) for v in unique_vals[:5]])  # Primeros 5 valores únicos
                }
        
        # 3. Combinar todo de manera determinística
        dataset_signature = {
            'metadata': metadata,
            'statistics': stats
        }
        
        # 4. Generar hash
        signature_str = str(sorted(dataset_signature.items()))
        hash_obj = hashlib.md5(signature_str.encode())
        dataset_hash = hash_obj.hexdigest()[:8]
        
        return f"real_{dataset_hash}"
        
    except Exception as e:
        # Fallback al método anterior si hay error leyendo el CSV
        print(f"Warning: Could not analyze dataset content ({e}), using file-based ID")
        try:
            stat = os.stat(file_path)
            file_info = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
        except:
            file_info = file_path
        
        hash_obj = hashlib.md5(file_info.encode())
        dataset_hash = hash_obj.hexdigest()[:8]
        return f"real_{dataset_hash}"

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

def generate_synthetic_data(params, output_file=None, base_datasets_dir=None):
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
    output_file : str, optional
        Name of the output CSV file (if None, will use default naming)
    base_datasets_dir : str, optional
        Base directory for datasets (if None, will use script directory)
        
    Returns:
    --------
    data_info : dict
        Dictionary with information about the generated data and paths
    """
    # Generate unique dataset ID
    dataset_id = generate_dataset_id(params, "synth")
    
    # Set up directory structure
    if base_datasets_dir is None:
        base_datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    
    # Create dataset-specific folder: datasets/IDs/{dataset_id}/
    dataset_dir = os.path.join(base_datasets_dir, "IDs", dataset_id)
    ensure_dir_exists(dataset_dir)
    
    # Set output file path
    if output_file is None:
        output_file = os.path.join(dataset_dir, f"{dataset_id}_dataset.csv")
    else:
        # If output_file is provided, put it in the dataset_dir
        output_file = os.path.join(dataset_dir, os.path.basename(output_file))
    
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
        "dataset_id": dataset_id,
        "dataset_dir": dataset_dir,
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