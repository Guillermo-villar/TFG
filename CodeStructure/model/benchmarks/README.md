# Benchmarks Directory

This directory stores benchmark results comparing CPU vs GPU performance for ML experiments.

## Folder Structure

Each benchmark run creates a folder with the following naming convention:

```
{system_id}_{device_mode}_{timestamp}/
```

Where:
- `system_id`: A sanitized identifier based on your system's CPU and GPU
- `device_mode`: Either `cpu`, `gpu`, or `comparison`
- `timestamp`: The date and time of the benchmark run

## Contents of Each Benchmark Folder

Each benchmark folder contains:

1. **`system_info.json`**: Complete system information including:
   - Operating system details
   - CPU information
   - GPU information (if available)
   - PyTorch and CUDA versions

2. **`benchmark_results.json`**: Detailed benchmark results including:
   - Timing information for each run
   - Model configuration used
   - Accuracy metrics
   - Speedup calculations (if both CPU and GPU were tested)

3. **`summary.txt`**: A human-readable summary of the benchmark results

## Running Benchmarks

To run a benchmark comparison, use the `benchmark_runner.py` script:

```bash
# Run with synthetic data
python benchmark_runner.py

# Run with your own dataset
python benchmark_runner.py --dataset path/to/your/data.csv

# See all available options
python benchmark_runner.py --help
```

## Example Output

```
==========================================================
BENCHMARK COMPARISON SUMMARY
==========================================================
CPU avg time: 15.2340s
GPU avg time: 3.4567s
GPU Speedup: 4.41x
==========================================================
```

## Notes

- GPU benchmarks require CUDA-compatible NVIDIA hardware
- If CUDA is not available, only CPU benchmarks will run
- Multiple runs are averaged for more reliable timing results
- The `.gitignore` file excludes benchmark results to keep the repository clean
