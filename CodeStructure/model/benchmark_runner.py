#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark Runner for GPU vs CPU Performance Comparison

This script runs ML experiments on both CPU and GPU (if available) and
compares the execution times, saving results to dedicated folders with
system specifications.
"""

import os
import sys
import time
import json
import logging
import datetime
import argparse

import numpy as np
import pandas as pd
import torch

# Add parent directory to import project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.dirname(script_dir))

from device_utils import (
    get_device, is_cuda_available, get_system_info, 
    save_system_info, print_device_info, get_benchmark_folder_name
)
from uc3m.labelswitching import LSEnsemble
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, confusion_matrix

# Setup logging
logging.basicConfig(
    format="%(asctime)-15s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Runs benchmarks comparing CPU and GPU performance for LSEnsemble models.
    """
    
    def __init__(self, output_base_dir: str = None):
        """
        Initialize the benchmark runner.
        
        Parameters:
        -----------
        output_base_dir : str
            Base directory for saving benchmark results.
            Defaults to 'benchmarks/' in the CodeStructure folder.
        """
        if output_base_dir is None:
            self.output_base_dir = os.path.join(script_dir, "benchmarks")
        else:
            self.output_base_dir = output_base_dir
        
        os.makedirs(self.output_base_dir, exist_ok=True)
    
    def run_single_benchmark(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        device: str,
        model_config: dict
    ) -> dict:
        """
        Run a single benchmark on the specified device.
        
        Parameters:
        -----------
        X_train, y_train : np.ndarray
            Training data
        X_test, y_test : np.ndarray
            Test data
        device : str
            Device to use ('cpu' or 'cuda')
        model_config : dict
            Configuration for the LSEnsemble model
            
        Returns:
        --------
        dict : Benchmark results including timing and metrics
        """
        logger.info(f"Running benchmark on device: {device}")
        
        # Update config with device
        config = model_config.copy()
        config['device'] = device
        
        # Initialize model
        model = LSEnsemble(**config)
        
        # Warm-up run (important for GPU)
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Time the training
        train_start = time.time()
        
        # Synchronize before timing (for accurate GPU timing)
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        model.fit(X_train, y_train)
        
        # Synchronize after training (for accurate GPU timing)
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        train_time = time.time() - train_start
        
        # Time the prediction
        predict_start = time.time()
        
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        y_pred = model.predict(X_test)
        
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        predict_time = time.time() - predict_start
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'device': device,
            'train_time_seconds': train_time,
            'predict_time_seconds': predict_time,
            'total_time_seconds': train_time + predict_time,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_features': X_train.shape[1],
            'model_config': config
        }
        
        logger.info(f"  Train time: {train_time:.4f}s")
        logger.info(f"  Predict time: {predict_time:.4f}s")
        logger.info(f"  Total time: {train_time + predict_time:.4f}s")
        logger.info(f"  Balanced Accuracy: {balanced_acc:.4f}")
        
        return results
    
    def run_comparison_benchmark(
        self,
        dataset_path: str = None,
        X_train: np.ndarray = None,
        y_train: np.ndarray = None,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
        model_config: dict = None,
        n_runs: int = 3,
        test_size: float = 0.2
    ) -> dict:
        """
        Run benchmarks on both CPU and GPU (if available) and compare results.
        
        Parameters:
        -----------
        dataset_path : str, optional
            Path to CSV dataset file. If provided, will load and split data.
        X_train, y_train, X_test, y_test : np.ndarray, optional
            Training and test data. Used if dataset_path is not provided.
        model_config : dict, optional
            Configuration for the LSEnsemble model.
        n_runs : int
            Number of runs to average for timing.
        test_size : float
            Fraction of data to use for testing if loading from CSV.
            
        Returns:
        --------
        dict : Comparison results
        """
        # Default model config
        if model_config is None:
            model_config = {
                'hidden_size': 30,
                'num_experts': 21,
                'alpha': 0.0,
                'beta': 0.0,
                'Q_RB_C': 2.0,
                'Q_RB_S': 1,
                'n_epoch': 50,
                'n_batch': 128,
                'lbfgs': False,
                'mode': 'random',
                'activation_fn': 'relu',
                'loss_fn': 'F1'  # Valid options: MSE, KL, BCE, BCE_logit, F1
            }
        
        # Load data from CSV if path provided
        if dataset_path is not None:
            logger.info(f"Loading dataset from {dataset_path}")
            df = pd.read_csv(dataset_path)
            
            # Determine target column
            if 'target' in df.columns:
                target_col = 'target'
            else:
                target_col = df.columns[-1]
            
            y = df[target_col].values
            X = df.drop(columns=[target_col]).values
            
            # Shuffle and split
            np.random.seed(42)
            indices = np.random.permutation(len(X))
            split_idx = int(len(X) * (1 - test_size))
            
            train_idx = indices[:split_idx]
            test_idx = indices[split_idx:]
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
        
        # Update model config with input size
        model_config['input_size'] = X_train.shape[1]
        
        # Determine which devices to test
        devices_to_test = ['cpu']
        if is_cuda_available():
            devices_to_test.append('cuda')
        else:
            logger.warning("CUDA not available. Only CPU benchmarks will be run.")
        
        # Run benchmarks
        all_results = {}
        
        for device in devices_to_test:
            device_results = []
            
            for run in range(n_runs):
                logger.info(f"\n=== Run {run + 1}/{n_runs} on {device.upper()} ===")
                result = self.run_single_benchmark(
                    X_train, y_train, X_test, y_test,
                    device, model_config
                )
                device_results.append(result)
            
            # Aggregate results
            avg_train_time = np.mean([r['train_time_seconds'] for r in device_results])
            avg_predict_time = np.mean([r['predict_time_seconds'] for r in device_results])
            avg_total_time = np.mean([r['total_time_seconds'] for r in device_results])
            std_total_time = np.std([r['total_time_seconds'] for r in device_results])
            
            all_results[device] = {
                'individual_runs': device_results,
                'avg_train_time_seconds': avg_train_time,
                'avg_predict_time_seconds': avg_predict_time,
                'avg_total_time_seconds': avg_total_time,
                'std_total_time_seconds': std_total_time,
                'avg_accuracy': np.mean([r['accuracy'] for r in device_results]),
                'avg_balanced_accuracy': np.mean([r['balanced_accuracy'] for r in device_results]),
                'avg_f1_score': np.mean([r['f1_score'] for r in device_results]),
            }
        
        # Calculate speedup if both CPU and GPU results exist
        comparison = {
            'benchmark_time': datetime.datetime.now().isoformat(),
            'n_runs': n_runs,
            'devices_tested': devices_to_test,
            'results': all_results,
            'system_info': get_system_info()
        }
        
        if 'cpu' in all_results and 'cuda' in all_results:
            cpu_time = all_results['cpu']['avg_total_time_seconds']
            gpu_time = all_results['cuda']['avg_total_time_seconds']
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            comparison['speedup'] = {
                'gpu_vs_cpu': speedup,
                'cpu_time_seconds': cpu_time,
                'gpu_time_seconds': gpu_time
            }
            logger.info(f"\n{'='*60}")
            logger.info(f"BENCHMARK COMPARISON SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"CPU avg time: {cpu_time:.4f}s")
            logger.info(f"GPU avg time: {gpu_time:.4f}s")
            logger.info(f"GPU Speedup: {speedup:.2f}x")
            logger.info(f"{'='*60}\n")
        
        return comparison
    
    def save_results(self, results: dict, device_mode: str) -> str:
        """
        Save benchmark results to a folder named with system specs.
        
        Parameters:
        -----------
        results : dict
            Benchmark results to save.
        device_mode : str
            'cpu', 'gpu', or 'comparison'
            
        Returns:
        --------
        str : Path to the saved results folder.
        """
        folder_name = get_benchmark_folder_name(device_mode)
        output_dir = os.path.join(self.output_base_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save system info
        save_system_info(output_dir)
        
        # Save benchmark results
        results_path = os.path.join(output_dir, "benchmark_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create a summary text file
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("BENCHMARK SUMMARY\n")
            f.write("="*60 + "\n")
            f.write(f"Time: {results.get('benchmark_time', 'N/A')}\n")
            f.write(f"Number of runs: {results.get('n_runs', 'N/A')}\n")
            f.write(f"Devices tested: {results.get('devices_tested', [])}\n\n")
            
            for device, device_results in results.get('results', {}).items():
                f.write(f"\n--- {device.upper()} Results ---\n")
                f.write(f"Avg Train Time: {device_results.get('avg_train_time_seconds', 'N/A'):.4f}s\n")
                f.write(f"Avg Predict Time: {device_results.get('avg_predict_time_seconds', 'N/A'):.4f}s\n")
                f.write(f"Avg Total Time: {device_results.get('avg_total_time_seconds', 'N/A'):.4f}s\n")
                f.write(f"Std Total Time: {device_results.get('std_total_time_seconds', 'N/A'):.4f}s\n")
                f.write(f"Avg Balanced Accuracy: {device_results.get('avg_balanced_accuracy', 'N/A'):.4f}\n")
            
            if 'speedup' in results:
                f.write("\n--- SPEEDUP ---\n")
                f.write(f"GPU vs CPU: {results['speedup']['gpu_vs_cpu']:.2f}x\n")
        
        logger.info(f"Results saved to: {output_dir}")
        return output_dir


def generate_synthetic_data(n_samples: int = 5000, n_features: int = 10, 
                            imbalance_ratio: float = 0.1) -> tuple:
    """
    Generate synthetic binary classification data with imbalance.
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples
    n_features : int
        Number of features
    imbalance_ratio : float
        Ratio of minority class (default 0.1 = 10% minority)
        
    Returns:
    --------
    tuple : (X_train, y_train, X_test, y_test)
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Calculate weights for imbalanced dataset
    minority_weight = imbalance_ratio
    majority_weight = 1 - imbalance_ratio
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features - 2,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[majority_weight, minority_weight],
        random_state=42,
        flip_y=0.01
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, y_train, X_test, y_test


def main():
    """Main entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description='Run GPU vs CPU performance benchmarks for ML experiments'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        help='Path to CSV dataset file'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for benchmark results'
    )
    parser.add_argument(
        '--n-runs', '-n',
        type=int,
        default=3,
        help='Number of benchmark runs to average'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=5000,
        help='Number of samples for synthetic data (if no dataset provided)'
    )
    parser.add_argument(
        '--n-features',
        type=int,
        default=10,
        help='Number of features for synthetic data'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'gpu', 'both'],
        default='both',
        help='Device to benchmark (cpu, gpu, or both)'
    )
    parser.add_argument(
        '--info-only',
        action='store_true',
        help='Only print device information and exit'
    )
    
    args = parser.parse_args()
    
    # Print device info
    print_device_info()
    
    if args.info_only:
        return
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(output_base_dir=args.output_dir)
    
    # Get data
    if args.dataset:
        logger.info(f"Using dataset: {args.dataset}")
        results = runner.run_comparison_benchmark(
            dataset_path=args.dataset,
            n_runs=args.n_runs
        )
    else:
        logger.info(f"Generating synthetic data: {args.n_samples} samples, {args.n_features} features")
        X_train, y_train, X_test, y_test = generate_synthetic_data(
            n_samples=args.n_samples,
            n_features=args.n_features
        )
        results = runner.run_comparison_benchmark(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_runs=args.n_runs
        )
    
    # Determine device mode for folder naming
    if args.device == 'both' and is_cuda_available():
        device_mode = 'comparison'
    elif args.device == 'gpu' and is_cuda_available():
        device_mode = 'gpu'
    else:
        device_mode = 'cpu'
    
    # Save results
    output_dir = runner.save_results(results, device_mode)
    
    print(f"\nBenchmark complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
