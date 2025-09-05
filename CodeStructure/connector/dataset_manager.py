import os
import sys
import yaml
import json
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to allow imports from the main project folder
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model import data_generator
from model.test_LSEnsemble import load_study_progress

def get_dataset_status(config_path, data_source="synthetic", csv_path=None, current_level=1):
    """
    Get the status of a dataset (new, in progress, or completed)
    Returns a dictionary with detailed status information including ETA
    Enhanced to support both binary and multiclass experiments
    """
    try:
        # Load current config to get capilaridad configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        current_capilaridad_config = config.get("capilaridad_levels", {}).get(current_level, {})
        
        # Generate dataset ID based on parameters
        if data_source == "synthetic":
            params = config.get("data", {}).get("params", {})
            dataset_id = data_generator.generate_dataset_id(params, "synth")
        else:
            # For real data, always use the content-based hash generation
            if not csv_path or not os.path.exists(csv_path):
                return {
                    "status": "error",
                    "message": f"CSV file path not provided or file does not exist: {csv_path}",
                    "current_capilaridad_config": current_capilaridad_config,
                    "current_level": current_level,
                    "experiment_type": "unknown"
                }
            dataset_id = data_generator.generate_real_data_id(csv_path)
        
        # First check for multiclass experiment
        dataset_paths = data_generator.get_dataset_directory_structure(dataset_id)
        multiclass_status = get_multiclass_status(dataset_paths, dataset_id, current_capilaridad_config, current_level)
        
        if multiclass_status:
            # This is a multiclass experiment, return multiclass status
            return multiclass_status
        
        # If not multiclass, continue with binary experiment logic
        # Get dataset directory structure
        dataset_paths = data_generator.get_dataset_directory_structure(dataset_id)
        best_runner_file = dataset_paths["best_runner_json"]
        
        # Check if best_runner file exists (binary experiment)
        if not os.path.exists(best_runner_file):
            return {
                "status": "new",
                "dataset_id": dataset_id,
                "message": "Dataset has not been studied before",
                "current_capilaridad_config": current_capilaridad_config,
                "current_level": current_level,
                "experiment_type": "binary"
            }
        
        # Load study progress
        study_data = load_study_progress(best_runner_file)
        
        if not study_data or "execution_state" not in study_data:
            # Old format or corrupted file
            return {
                "status": "existing_old",
                "dataset_id": dataset_id,
                "message": "Dataset has old format study data",
                "best_runner": study_data.get("best_runner") if study_data else None,
                "current_capilaridad_config": current_capilaridad_config,
                "current_level": current_level,
                "experiment_type": "binary"
            }
        
        exec_state = study_data["execution_state"]
        best_runner = study_data["best_runner"]
        capilaridad_config = study_data.get("capilaridad_config", {})
        
        # Determine status based on execution state
        stage1_complete = exec_state.get("stage1_completed", False)
        stage1_total = exec_state.get("stage1_total", 0)
        stage1_done = exec_state.get("stage1_position", 0)
        stage2_total = exec_state.get("stage2_total", 0)
        stage2_done = exec_state.get("stage2_position", 0)
        
        # Get previous capilaridad configuration
        previous_capilaridad_config = capilaridad_config
        previous_level = previous_capilaridad_config.get("level", "unknown")
        
        # Get timing and ETA information
        timing_info = get_timing_and_eta(dataset_paths, stage1_done, stage1_total, 
                                             stage2_done, stage2_total, stage1_complete)
        
        status_info = {
            "status": "existing",
            "dataset_id": dataset_id,
            "stage1_completed": stage1_complete,
            "stage1_progress": f"{stage1_done}/{stage1_total}",
            "stage2_progress": f"{stage2_done}/{stage2_total}",
            "current_stage": exec_state.get("current_stage", 1),
            "last_updated": exec_state.get("last_updated", "Unknown"),
            "best_runner": best_runner,
            "study_mode": exec_state.get("study_mode", "full_study"),
            "capilaridad_config": capilaridad_config,
            "previous_capilaridad_config": previous_capilaridad_config,
            "current_capilaridad_config": current_capilaridad_config,
            "previous_level": previous_level,
            "current_level": current_level,
            "best_metric_so_far": exec_state.get("best_metric_so_far"),
            "experiment_type": "binary",
            **timing_info  # Add timing information
        }
        
        # Determine overall status
        if stage1_complete and stage2_done >= stage2_total and stage2_total > 0:
            status_info["overall_status"] = "complete"
            status_info["message"] = "Study is complete"
        elif stage1_complete:
            status_info["overall_status"] = "stage2_in_progress"
            status_info["message"] = "Stage 1 complete, Stage 2 in progress"
        elif stage1_total > 0:
            status_info["overall_status"] = "stage1_in_progress"
            status_info["message"] = "Stage 1 in progress"
        else:
            status_info["overall_status"] = "unknown"
            status_info["message"] = "Unknown status"
        
        return status_info
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error checking dataset status: {str(e)}",
            "current_capilaridad_config": current_capilaridad_config if 'current_capilaridad_config' in locals() else {},
            "current_level": current_level,
            "experiment_type": "unknown"
        }

def get_multiclass_status(dataset_paths, dataset_id, current_capilaridad_config, current_level):
    """
    Check for multiclass experiment status and return appropriately formatted progress.
    """
    try:
        # Check for multiclass progress file
        multiclass_progress_file = os.path.join(dataset_paths["dataset_dir"], f"multiclass_progress_{dataset_id}.json")
        multiclass_best_runner_file = os.path.join(dataset_paths["dataset_dir"], f"multiclass_best_runner_{dataset_id}.json")
        
        if not os.path.exists(multiclass_progress_file):
            return None  # No multiclass experiment
        
        # Load multiclass progress
        with open(multiclass_progress_file, 'r') as f:
            multiclass_data = json.load(f)
        
        # Load multiclass best runner if available
        multiclass_best_runner = None
        if os.path.exists(multiclass_best_runner_file):
            with open(multiclass_best_runner_file, 'r') as f:
                multiclass_best_runner = json.load(f)
        
        # Parse multiclass progress
        dichotomies = multiclass_data.get("dichotomies", {})
        total_dichotomies = len(dichotomies)
        completed_dichotomies = sum(1 for d in dichotomies.values() if d.get("status") == "completed")
        in_progress_dichotomies = sum(1 for d in dichotomies.values() if d.get("status") == "in_progress")
        
        stage1_progress = f"{completed_dichotomies}/{total_dichotomies}"
        stage2_progress = "0/0"
        
        if completed_dichotomies == total_dichotomies and total_dichotomies > 0:
            overall_status, message, stage1_complete, current_stage = "complete", f"Multiclass study is complete ({completed_dichotomies}/{total_dichotomies} dichotomies)", True, 2
        elif in_progress_dichotomies > 0:
            overall_status, message, stage1_complete, current_stage = "stage1_in_progress", f"Multiclass study in progress ({completed_dichotomies}/{total_dichotomies} dichotomies completed)", False, 1
        elif completed_dichotomies > 0:
            overall_status, message, stage1_complete, current_stage = "stage1_in_progress", f"Multiclass study resumed ({completed_dichotomies}/{total_dichotomies} dichotomies completed)", False, 1
        else:
            overall_status, message, stage1_complete, current_stage = "new", f"Multiclass study ready to start ({total_dichotomies} dichotomies)", False, 1
        
        timing_info = get_multiclass_timing_and_eta(dataset_paths, completed_dichotomies, total_dichotomies)
        
        best_metric_so_far = multiclass_best_runner.get("best_metric_so_far") if multiclass_best_runner else None
        
        return {
            "status": "existing", "dataset_id": dataset_id, "stage1_completed": stage1_complete,
            "stage1_progress": stage1_progress, "stage2_progress": stage2_progress, "current_stage": current_stage,
            "last_updated": multiclass_data.get("last_updated", "Unknown"),
            "best_runner": multiclass_best_runner.get("best_runner_config") if multiclass_best_runner else None,
            "study_mode": "multiclass", "capilaridad_config": {"level": current_level},
            "previous_capilaridad_config": {"level": current_level}, "current_capilaridad_config": current_capilaridad_config,
            "previous_level": current_level, "current_level": current_level, "best_metric_so_far": best_metric_so_far,
            "experiment_type": "multiclass", "overall_status": overall_status, "message": message,
            "multiclass_info": {
                "strategy": multiclass_data.get("strategy", "unknown").upper(),
                "n_classes": multiclass_data.get("n_classes", "unknown"),
                "class_names": multiclass_data.get("class_names", [])[:5],
                "total_dichotomies": total_dichotomies, "completed_dichotomies": completed_dichotomies,
                "in_progress_dichotomies": in_progress_dichotomies,
                "current_dichotomy": multiclass_data.get("summary", {}).get("current_dichotomy"),
                "current_dichotomy_index": multiclass_data.get("summary", {}).get("current_dichotomy_index"),
                "dichotomy_details": dichotomies
            }, **timing_info
        }
    except Exception:
        return None

def get_multiclass_timing_and_eta(dataset_paths, completed_dichotomies, total_dichotomies):
    """
    Calculate timing statistics and ETA for multiclass experiments.
    """
    timing_info = {
        "avg_time_per_run": 0, "total_time_elapsed": 0, "eta_seconds": None,
        "eta_formatted": "Unknown", "total_runs_completed": completed_dichotomies,
        "runs_per_hour": 0, "remaining_runs": total_dichotomies - completed_dichotomies,
        "current_progress": completed_dichotomies, "total_stage_runs": total_dichotomies,
        "stage_name": "Dichotomies", "dataset_size": 0, "using_estimate": True
    }
    try:
        actual_times = []
        dataset_dir = dataset_paths["dataset_dir"]
        
        for i in range(1, completed_dichotomies + 1):
            dichotomy_run_dir = os.path.join(dataset_dir, f"dichotomy_{i:02d}_run")
            if os.path.exists(dichotomy_run_dir):
                dichotomy_results_csv = os.path.join(dichotomy_run_dir, "test_results.csv")
                if os.path.exists(dichotomy_results_csv):
                    try:
                        df = pd.read_csv(dichotomy_results_csv)
                        if 'time_taken' in df.columns:
                            actual_times.append(df['time_taken'].sum())
                    except Exception: continue
        
        if actual_times:
            avg_time_per_dichotomy = np.mean(actual_times)
            remaining_dichotomies = total_dichotomies - completed_dichotomies
            if remaining_dichotomies > 0:
                eta_seconds = remaining_dichotomies * avg_time_per_dichotomy
                if eta_seconds < 3600: eta_formatted = f"{int(eta_seconds / 60)}m"
                elif eta_seconds < 86400: eta_formatted = f"{int(eta_seconds / 3600)}h {int((eta_seconds % 3600) / 60)}m"
                else: eta_formatted = f"{int(eta_seconds / 86400)}d {int((eta_seconds % 86400) / 3600)}h"
                timing_info.update({"eta_seconds": eta_seconds, "eta_formatted": eta_formatted})
            
            timing_info.update({
                "avg_time_per_run": round(avg_time_per_dichotomy, 2),
                "total_time_elapsed": round(sum(actual_times), 2),
                "runs_per_hour": round(3600 / avg_time_per_dichotomy if avg_time_per_dichotomy > 0 else 0, 2),
                "using_estimate": False
            })
        else:
            remaining_dichotomies = total_dichotomies - completed_dichotomies
            if remaining_dichotomies > 0:
                eta_seconds = remaining_dichotomies * 600
                hours, rem = divmod(eta_seconds, 3600)
                minutes, _ = divmod(rem, 60)
                eta_formatted = f"~{int(hours)}h {int(minutes)}m" if hours > 0 else f"~{int(minutes)}m"
                timing_info.update({"avg_time_per_run": 600, "eta_seconds": eta_seconds, "eta_formatted": eta_formatted, "runs_per_hour": 0.1, "using_estimate": True})
    except Exception:
        remaining_dichotomies = total_dichotomies - completed_dichotomies
        if remaining_dichotomies > 0:
            timing_info.update({"avg_time_per_run": 600, "eta_formatted": f"~{remaining_dichotomies * 10}m", "using_estimate": True})
    return timing_info

def get_timing_and_eta(dataset_paths, stage1_done, stage1_total, stage2_done, stage2_total, stage1_complete):
    """
    Calculate timing statistics and ETA based on dataset size and actual run times
    """
    timing_info = {"avg_time_per_run": 0, "total_time_elapsed": 0, "eta_seconds": None, "eta_formatted": "Unknown", "total_runs_completed": 0, "runs_per_hour": 0}
    try:
        dataset_csv = dataset_paths["processed_csv"]
        dataset_size = len(pd.read_csv(dataset_csv)) if os.path.exists(dataset_csv) else 0
        
        estimated_time_per_run = 30.0 if dataset_size > 1000 else (15.0 if dataset_size > 500 else 10.0)
        
        main_results_csv = os.path.join(dataset_paths["dataset_dir"], "test_results.csv")
        actual_times = []
        if os.path.exists(main_results_csv):
            try:
                df = pd.read_csv(main_results_csv)
                if 'time_taken' in df.columns:
                    all_times = df['time_taken'].dropna().tolist()
                    if stage1_complete and stage2_done > 0 and len(all_times) > stage1_total:
                        actual_times = all_times[stage1_total:]
                    else:
                        actual_times = all_times
            except Exception: pass
        
        if actual_times:
            avg_time = np.mean(actual_times)
            weight_actual = min(len(actual_times) / (3.0 if stage1_complete else 10.0), 1.0)
            weighted_avg_time = (1 - weight_actual) * estimated_time_per_run + weight_actual * avg_time
        else:
            weighted_avg_time = estimated_time_per_run
        
        if stage1_complete:
            remaining_runs, current_progress, total_stage_runs, stage_name = stage2_total - stage2_done, stage2_done, stage2_total, "Stage 2"
        else:
            remaining_runs, current_progress, total_stage_runs, stage_name = stage1_total - stage1_done, stage1_done, stage1_total, "Stage 1"
        
        if remaining_runs > 0 and weighted_avg_time > 0:
            eta_seconds = remaining_runs * weighted_avg_time
            if eta_seconds < 60: eta_formatted = f"{int(eta_seconds)}s"
            elif eta_seconds < 3600: eta_formatted = f"{int(eta_seconds / 60)}m {int(eta_seconds % 60)}s"
            else: eta_formatted = f"{int(eta_seconds / 3600)}h {int((eta_seconds % 3600) / 60)}m"
            timing_info.update({"eta_seconds": eta_seconds, "eta_formatted": eta_formatted})
        
        timing_info.update({
            "avg_time_per_run": round(weighted_avg_time, 2), "total_time_elapsed": round(sum(actual_times), 2),
            "runs_per_hour": round(3600 / weighted_avg_time if weighted_avg_time > 0 else 0, 1),
            "remaining_runs": remaining_runs, "current_progress": current_progress, "total_stage_runs": total_stage_runs,
            "stage_name": stage_name, "dataset_size": dataset_size, "using_estimate": len(actual_times) < (3 if stage1_complete else 10)
        })
    except Exception:
        timing_info.update({"avg_time_per_run": 15.0, "eta_formatted": "Calculating...", "using_estimate": True})
    return timing_info

def delete_dataset_progress(config_path, data_source="synthetic", csv_path=None):
    """
    Delete all progress data for a dataset, allowing fresh start.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if data_source == "synthetic":
            params = config.get("data", {}).get("params", {})
            dataset_id = data_generator.generate_dataset_id(params, "synth")
        else:
            dataset_id = data_generator.generate_real_data_id(csv_path) if csv_path and os.path.exists(csv_path) else data_generator.generate_dataset_id({"file_path": csv_path}, "real")
        
        dataset_paths = data_generator.get_dataset_directory_structure(dataset_id)
        dataset_dir = dataset_paths["dataset_dir"]
        
        deleted_items = []
        multiclass_progress_file = os.path.join(dataset_dir, f"multiclass_progress_{dataset_id}.json")
        is_multiclass = os.path.exists(multiclass_progress_file)

        if is_multiclass:
            if os.path.exists(multiclass_progress_file):
                os.remove(multiclass_progress_file); deleted_items.append("Multiclass progress file")
            multiclass_best_runner_file = os.path.join(dataset_dir, f"multiclass_best_runner_{dataset_id}.json")
            if os.path.exists(multiclass_best_runner_file):
                os.remove(multiclass_best_runner_file); deleted_items.append("Multiclass best runner configuration")
            
            dichotomy_dirs_deleted = 0
            if os.path.exists(dataset_dir):
                for item in os.listdir(dataset_dir):
                    item_path = os.path.join(dataset_dir, item)
                    if os.path.isdir(item_path) and item.startswith("dichotomy_") and item.endswith("_run"):
                        shutil.rmtree(item_path); dichotomy_dirs_deleted += 1
            if dichotomy_dirs_deleted > 0: deleted_items.append(f"Dichotomy run directories ({dichotomy_dirs_deleted})")
        
        best_runner_file = dataset_paths["best_runner_json"]
        if os.path.exists(best_runner_file):
            os.remove(best_runner_file); deleted_items.append("Best runner configuration")
        
        runs_dir = dataset_paths["runs_dir"]
        if os.path.exists(runs_dir):
            shutil.rmtree(runs_dir); deleted_items.append("All run history")
        
        main_results_csv = os.path.join(dataset_dir, "test_results.csv")
        if os.path.exists(main_results_csv):
            os.remove(main_results_csv); deleted_items.append("Main results file")
        
        if deleted_items:
            return {"success": True, "message": f"Successfully deleted {'multiclass' if is_multiclass else 'binary'} experiment progress: {', '.join(deleted_items)}", "dataset_id": dataset_id}
        else:
            return {"success": True, "message": "No progress data found to delete", "dataset_id": dataset_id}
            
    except Exception as e:
        return {"success": False, "message": f"Error deleting progress: {str(e)}", "dataset_id": locals().get('dataset_id', "unknown")}

def get_experiment_folder_path(config_path, data_source="synthetic", csv_path=None):
    """Get the path to the experiment folder for report generation"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if data_source == "synthetic":
            params = config.get("data", {}).get("params", {})
            dataset_id = data_generator.generate_dataset_id(params, "synth")
        else:
            dataset_id = data_generator.generate_real_data_id(csv_path) if csv_path and os.path.exists(csv_path) else data_generator.generate_dataset_id({"file_path": csv_path}, "real")
        
        return data_generator.get_dataset_directory_structure(dataset_id)["dataset_dir"]
    except Exception:
        return None
