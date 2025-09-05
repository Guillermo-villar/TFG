import os
import sys
import json
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, cohen_kappa_score
from imblearn.metrics import geometric_mean_score, sensitivity_score

# Add parent directory to allow imports from the main project folder
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model import data_generator
from model.csv_preloading import CSVPreprocessor, create_multiclass_dichotomies
from model.uc3m.labelswitching import LSEnsemble

def generate_experiment_report(parent_dir, config_path, data_source="synthetic", csv_path=None, current_level=1, gui_root=None, language: str = "en", experiment_runner=None):
    """
    Generate and show experiment report popup automatically after completion.
    """
    try:
        from report_generation.pdf_preview_popup import show_report_popup
        
        experiment_path = get_experiment_folder_path(config_path, data_source, csv_path)
        
        if not experiment_path or not os.path.exists(experiment_path):
            raise Exception("No experiment folder found to generate report")
        
        show_report_popup(experiment_path, language=language, parent=gui_root, experiment_runner=experiment_runner)
        
    except Exception as e:
        raise Exception(f"Error generating experiment report: {str(e)}")

def diagnose_path_mismatch(original_csv_path):
    """
    Diagnose the path mismatch issue for multiclass experiments.
    """
    try:
        id_from_original = data_generator.generate_real_data_id(original_csv_path)
        original_folder_path = data_generator.get_dataset_directory_structure(id_from_original)["dataset_dir"]
        
        preprocessor = CSVPreprocessor(gui_enabled=False)
        multiclass_candidates = preprocessor.detect_multiclass(original_csv_path)
        
        if not multiclass_candidates:
            return {"success": False, "error": "No multiclass candidates found"}
        
        dichotomy_result = create_multiclass_dichotomies(original_csv_path, multiclass_candidates[0], strategy="ova")
        
        if not dichotomy_result['success']:
            return {"success": False, "error": f"Failed to create dichotomies: {dichotomy_result.get('error', 'Unknown error')}"}
        
        first_dichotomy_path = dichotomy_result['dichotomy_files'][0]
        id_from_dichotomy = data_generator.generate_real_data_id(first_dichotomy_path)
        dichotomy_folder_path = data_generator.get_dataset_directory_structure(id_from_dichotomy)["dataset_dir"]
        
        for temp_file in dichotomy_result['dichotomy_files']:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return {
            "success": True, "id_from_original": id_from_original, "id_from_dichotomy": id_from_dichotomy,
            "original_folder_path": original_folder_path, "dichotomy_folder_path": dichotomy_folder_path,
            "dichotomy_filepath": first_dichotomy_path, "ids_match": id_from_original == id_from_dichotomy
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def reconstitute_multiclass_models(dataset_id):
    """
    Orchestrates the reconstitution of multiclass models.
    """
    try:
        dataset_paths = data_generator.get_dataset_directory_structure(dataset_id)
        dataset_dir = dataset_paths["dataset_dir"]
        if not os.path.exists(dataset_dir):
            return {"success": False, "message": f"Dataset directory not found: {dataset_dir}"}

        multiclass_progress_file = os.path.join(dataset_dir, f"multiclass_progress_{dataset_id}.json")
        if not os.path.exists(multiclass_progress_file):
            return {"success": False, "message": "Multiclass progress file not found"}

        with open(multiclass_progress_file, 'r') as f:
            multiclass_data = json.load(f)
        
        strategy = multiclass_data.get("strategy", "ova")
        n_classes = multiclass_data.get("n_classes", 0)
        class_names = multiclass_data.get("class_names", [])
        M_ecoc_matrix = np.array(multiclass_data.get("code_matrix", []))

        # Look for any CSV file that is not a dichotomy or processed file
        original_csv_path = None
        for f in os.listdir(dataset_dir):
            if f.endswith('.csv') and 'dichotomy' not in f and 'processed' not in f:
                original_csv_path = os.path.join(dataset_dir, f)
                break
        
        if not original_csv_path:
            return {"success": False, "message": "Original dataset file not found"}

        df = pd.read_csv(original_csv_path)
        X = df.iloc[:, :-1].values
        y_original = df.iloc[:, -1].values
        
        class_dict = {class_names[i]: i + 1 for i in range(len(class_names))}
        y = np.array([class_dict.get(str(label), class_dict.get(label, 1)) for label in y_original])
        class_labels = np.array(sorted(class_dict.values()))

        best_configs = {}
        for d_dir in sorted([d for d in os.listdir(dataset_dir) if d.startswith('dichotomy_') and d.endswith('_run')]):
            d_path = os.path.join(dataset_dir, d_dir)
            d_name = d_dir.replace('_run', '')
            best_runner_file = os.path.join(d_path, f"best_runner_{d_name}.json")
            if os.path.exists(best_runner_file):
                with open(best_runner_file, 'r') as f:
                    best_configs[d_name] = json.load(f).get('best_runner', {})
            else:
                return {"success": False, "message": f"Best runner file not found for {d_name}"}

        if not best_configs:
            return {"success": False, "message": "No best configurations found"}

        n_simulations = 10
        metrics_sim = {k: [] for k in ["acc", "bal_acc", "kappa", "geom_mean", "sensitivity"]}

        for k_simu in range(n_simulations):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 + k_simu, stratify=y)
            scaler = StandardScaler().fit(X_train)
            X_train_n, X_test_n = scaler.transform(X_train), scaler.transform(X_test)
            
            Ye_pred = np.zeros((X_test.shape[0], M_ecoc_matrix.shape[1]))

            for j_dic in range(M_ecoc_matrix.shape[1]):
                try:
                    ecoc_column = M_ecoc_matrix[:, j_dic]
                    ye_train = np.array([ecoc_column[class_idx - 1] for class_idx in y_train])
                    
                    mask = ye_train != 0
                    X_train_filtered, ye_train_filtered = X_train_n[mask], ye_train[mask]

                    if len(np.unique(ye_train_filtered)) < 2:
                        Ye_pred[:, j_dic] = np.unique(ye_train_filtered)[0] if len(np.unique(ye_train_filtered)) > 0 else 1.0
                        continue

                    dichotomy_name = f"dichotomy_{j_dic+1:02d}"
                    if dichotomy_name in best_configs:
                        model = _instantiate_model(best_configs[dichotomy_name])
                        model.fit(X_train_filtered.astype(np.float32), ye_train_filtered.astype(np.float32))
                        Ye_pred[:, j_dic] = model.predict(X_test_n.astype(np.float32)).flatten()
                except Exception as e:
                    Ye_pred[:, j_dic] = np.random.choice([-1.0, 1.0], size=X_test.shape[0])

            y_pred_MC = _decode_ecoc_predictions(Ye_pred, M_ecoc_matrix, class_labels)
            
            metrics_sim["acc"].append(accuracy_score(y_test, y_pred_MC))
            metrics_sim["bal_acc"].append(balanced_accuracy_score(y_test, y_pred_MC))
            metrics_sim["kappa"].append(cohen_kappa_score(y_test, y_pred_MC))
            try:
                metrics_sim["geom_mean"].append(geometric_mean_score(y_test, y_pred_MC, average='weighted'))
                metrics_sim["sensitivity"].append(sensitivity_score(y_test, y_pred_MC, average='weighted'))
            except:
                metrics_sim["geom_mean"].append(0.0)
                metrics_sim["sensitivity"].append(0.0)

        final_metrics = {f"avg_{k}": np.mean(v) for k, v in metrics_sim.items()}
        final_metrics.update({f"std_{k}": np.std(v) for k, v in metrics_sim.items()})

        reconstitution_results = {
            "dataset_id": dataset_id, "strategy": strategy, "n_classes": n_classes, "class_names": class_names,
            "ecoc_matrix": M_ecoc_matrix.tolist(), "n_simulations": n_simulations, "final_metrics": final_metrics,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        with open(os.path.join(dataset_dir, f"reconstitution_results_{dataset_id}.json"), 'w') as f:
            json.dump(reconstitution_results, f, indent=2)

        return {"success": True, "message": f"Reconstitution completed. Avg Acc: {final_metrics['avg_acc']:.4f}", "results": reconstitution_results}

    except Exception as e:
        return {"success": False, "message": f"Error during reconstitution: {str(e)}"}

def _instantiate_model(best_config):
    """Instantiate the LSEnsemble model from configuration."""
    try:
        model_config = best_config.copy()
        return LSEnsemble(**model_config)
    except Exception as e:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=42)

def _decode_ecoc_predictions(Y_pred, M_ecoc, class_labels):
    """Decode ECOC predictions to multiclass labels."""
    y_pred_multiclass = np.zeros(Y_pred.shape[0], dtype=int)
    for i in range(Y_pred.shape[0]):
        distances = np.sum(Y_pred[i, :] != M_ecoc, axis=1)
        predicted_class_idx = np.argmin(distances)
        y_pred_multiclass[i] = class_labels[predicted_class_idx]
    return y_pred_multiclass

def get_experiment_folder_path(config_path, data_source="synthetic", csv_path=None):
    """Get the path to the experiment folder."""
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
