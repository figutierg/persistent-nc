 
import json
import os


def save_run_parameters(exp_config_dict):
    """
    Saves the provided experiment configuration dictionary to a JSON file
    for documentation and reproducibility.
    """
    # It now gets BOTH the experiment name AND the directory from the dictionary.
    exp_name = exp_config_dict["EXPERIMENT_NAME"]
    base_snapshot_dir = exp_config_dict["SNAPSHOT_DIR"]

    # The rest of the function works as before.
    params_dir = os.path.join(base_snapshot_dir, "run_parameters")
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)

    params_path = os.path.join(params_dir, f"{exp_name}_params.json")

    with open(params_path, 'w') as f:
        json.dump(exp_config_dict, f, indent=4)

    print(f"Saved run parameters to {params_path}")