#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Device Utilities for GPU/CPU Detection and Management

This module provides utilities for detecting hardware capabilities,
selecting compute devices (CPU/CUDA GPU), and collecting system specifications.
"""

import os
import platform
import json
import datetime
import logging
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the appropriate compute device based on preference and availability.
    
    Parameters:
    -----------
    prefer_gpu : bool
        If True, prefer GPU if available. If False, use CPU.
        
    Returns:
    --------
    torch.device
        The selected compute device.
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if prefer_gpu and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
        else:
            logger.info("Using CPU device")
    return device


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def get_gpu_info() -> Optional[Dict]:
    """
    Get detailed GPU information if CUDA is available.
    
    Returns:
    --------
    dict or None
        Dictionary with GPU information, or None if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return None
    
    gpu_info = {
        "name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "capability": torch.cuda.get_device_capability(0),
    }
    
    # Get memory info
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_info["total_memory_gb"] = round(total_memory / (1024**3), 2)
    except Exception as e:
        logger.warning(f"Could not get GPU memory info: {e}")
    
    return gpu_info


def get_cpu_info() -> Dict:
    """
    Get CPU information.
    
    Returns:
    --------
    dict
        Dictionary with CPU information.
    """
    cpu_info = {
        "processor": platform.processor() or "Unknown",
        "machine": platform.machine(),
        "physical_cores": os.cpu_count(),
    }
    
    # Try to get more detailed CPU info on Linux
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
                for line in cpuinfo.split("\n"):
                    if "model name" in line:
                        cpu_info["model_name"] = line.split(":")[1].strip()
                        break
        except Exception:
            pass
    
    return cpu_info


def get_system_info() -> Dict:
    """
    Get comprehensive system information.
    
    Returns:
    --------
    dict
        Dictionary with system information including OS, CPU, GPU (if available).
    """
    system_info = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "cpu": get_cpu_info(),
        "gpu": get_gpu_info(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    
    if torch.cuda.is_available():
        system_info["cuda_version"] = torch.version.cuda
    
    return system_info


def generate_system_id() -> str:
    """
    Generate a unique identifier based on system specs.
    
    Returns:
    --------
    str
        A sanitized string identifier for the system.
    """
    parts = []
    
    # Add OS info
    parts.append(platform.system().lower())
    
    # Add CPU info
    cpu_info = get_cpu_info()
    cpu_name = cpu_info.get("model_name", cpu_info.get("processor", "unknown_cpu"))
    # Sanitize the CPU name
    cpu_name = "".join(c if c.isalnum() else "_" for c in cpu_name)
    cpu_name = cpu_name[:30]  # Limit length
    parts.append(cpu_name)
    
    # Add GPU info if available
    if torch.cuda.is_available():
        gpu_info = get_gpu_info()
        gpu_name = gpu_info.get("name", "unknown_gpu")
        # Sanitize the GPU name
        gpu_name = "".join(c if c.isalnum() else "_" for c in gpu_name)
        gpu_name = gpu_name[:30]  # Limit length
        parts.append(gpu_name)
    
    return "_".join(filter(None, parts))


def get_benchmark_folder_name(device_mode: str) -> str:
    """
    Generate a folder name for benchmark results based on system specs and mode.
    
    Parameters:
    -----------
    device_mode : str
        Either "cpu" or "gpu"
        
    Returns:
    --------
    str
        Folder name in format: {system_id}_{device_mode}_{timestamp}
    """
    system_id = generate_system_id()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{system_id}_{device_mode}_{timestamp}"


def save_system_info(output_dir: str) -> str:
    """
    Save system information to a JSON file in the specified directory.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the system info file.
        
    Returns:
    --------
    str
        Path to the saved file.
    """
    system_info = get_system_info()
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, "system_info.json")
    with open(file_path, "w") as f:
        json.dump(system_info, f, indent=2)
    
    logger.info(f"System info saved to {file_path}")
    return file_path


def print_device_info():
    """Print detailed device information to console."""
    print("\n" + "="*60)
    print("DEVICE INFORMATION")
    print("="*60)
    
    print(f"\nPlatform: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    
    cpu_info = get_cpu_info()
    print(f"\nCPU: {cpu_info.get('model_name', cpu_info.get('processor', 'Unknown'))}")
    print(f"CPU Cores: {cpu_info.get('physical_cores', 'Unknown')}")
    
    if torch.cuda.is_available():
        gpu_info = get_gpu_info()
        print(f"\nCUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {gpu_info.get('name', 'Unknown')}")
        print(f"GPU Memory: {gpu_info.get('total_memory_gb', 'Unknown')} GB")
        print(f"CUDA Capability: {gpu_info.get('capability', 'Unknown')}")
    else:
        print(f"\nCUDA Available: No")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test the module
    print_device_info()
    
    print("Testing device selection:")
    cpu_device = get_device(prefer_gpu=False)
    print(f"CPU Device: {cpu_device}")
    
    gpu_device = get_device(prefer_gpu=True)
    print(f"GPU Device (preferred): {gpu_device}")
    
    print(f"\nSystem ID: {generate_system_id()}")
    print(f"Benchmark folder (CPU): {get_benchmark_folder_name('cpu')}")
    print(f"Benchmark folder (GPU): {get_benchmark_folder_name('gpu')}")
