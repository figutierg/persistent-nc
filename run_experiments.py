# run_experiments.py

import torch
import torch.nn as nn
import torch.optim as optim
import json
import config as global_config

from data_utils import get_dataset_info, get_transforms, get_dataloaders
from model_utils import create_model
from training import train
from utils import save_run_parameters
import os

def run_single_experiment(exp_config):
    """
    Runs a single, complete experiment based on a configuration dictionary.
    """
    device = torch.device(global_config.DEVICE)
    exp_name = exp_config["EXPERIMENT_NAME"]
    dataset_name = exp_config["DATASET_NAME"]
    snapshot_dir = exp_config["SNAPSHOT_DIR"]

    print(f"\n" + "=" * 60)
    print(f"--- Starting Experiment: {exp_name} ---")
    print(f"--- Saving all results to: {snapshot_dir} ---")
    print("=" * 60)

    save_run_parameters(exp_config)

    # --- CORRECTED SECTION ---
    # 1. Get the entire information dictionary for the dataset.

    dataset_info = get_dataset_info(dataset_name)
    train_transform, test_transform = get_transforms(
        input_size=dataset_info["input_size"],
        mean=dataset_info["mean"],
        std=dataset_info["std"]
    )

    # This logic is now much cleaner and more robust
    if exp_config["CLASSES_TO_USE"] == "all":
        num_classes = dataset_info["num_classes"]
        classes_to_use = list(range(num_classes))
    else:
        classes_to_use = exp_config["CLASSES_TO_USE"]
        num_classes = len(classes_to_use)
    # --- END OF CORRECTION ---


    train_loader, train_loader_eval, train_loader_nc, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        train_transform=train_transform,
        test_transform=test_transform,
        train_classes=classes_to_use,
        batch_size=exp_config["BATCH_SIZE"],
        eval_batch_size=exp_config.get("EVAL_BATCH_SIZE", exp_config["BATCH_SIZE"] * 2),
        num_workers=getattr(global_config, 'NUM_WORKERS', 4),
        pin_memory=getattr(global_config, 'PIN_MEMORY', True),
        nc_sample_size=exp_config["NC_SAMPLE_SIZE"] # Pass the new parameter
    )
    model = create_model(
        device,
        num_classes,
        exp_config["ARCHITECTURE_NAME"],
        dataset_info["model_adapter"]
    )

    start_epoch = 0
    if "RESUME_FROM_EPOCH" in exp_config and exp_config["RESUME_FROM_EPOCH"] > 0:
        start_epoch = exp_config["RESUME_FROM_EPOCH"]
        model_checkpoint_path = os.path.join(snapshot_dir, "models", f"{exp_name}_epoch_{start_epoch}.pth")

        if os.path.exists(model_checkpoint_path):
            print(f"--- Resuming training from epoch {start_epoch} ---")
            print(f"Loading model checkpoint from: {model_checkpoint_path}")
            model.load_state_dict(torch.load(model_checkpoint_path))
        else:
            print(
                f"WARNING: Checkpoint for epoch {start_epoch} not found at '{model_checkpoint_path}'. Starting from scratch.")
            start_epoch = 0  # Reset to 0 if checkpoint is not found

    optimizer = optim.SGD(
        model.parameters(),
        lr=exp_config["TRAIN_LR"],
        momentum=global_config.TRAIN_MOMENTUM,
        weight_decay=global_config.TRAIN_WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=global_config.SCHEDULER_MODE,
        factor=exp_config["SCHEDULER_FACTOR"],
        patience=exp_config["SCHEDULER_PATIENCE"],
        verbose=global_config.SCHEDULER_VERBOSE
    )

    train(
        model=model,
        start_epoch=start_epoch,
        train_loader=train_loader,
        train_loader_eval=train_loader_eval,
        train_loader_nc=train_loader_nc, # Pass the new loader
        test_loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        n_epochs=exp_config["N_EPOCHS"],
        snapshot_prefix=exp_name,
        snapshot_dir=snapshot_dir,
        pd_sample_size=exp_config["PD_SAMPLE_SIZE"],
        pd_max_edge_length=exp_config["PD_MAX_EDGE_LENGTH"],
        save_model_history=exp_config["SAVE_MODEL_HISTORY"]
    )
    print(f"--- Finished Experiment: {exp_name} ---")


if __name__ == '__main__':
    experiment_files = [
        'configs/resnet18_cifar100.json',
 #       'configs/densenet121_cifar100.json',
 #       'configs/resnet18_mnist.json',
 #       'configs/densenet121_mnist.json',
 #       'configs/resnet18_fmnist.json',
 #       'configs/densenet121_fmnist.json',
 #       'configs/resnet18_stl10.json',
 #       'configs/densenet121_stl10.json',
  #      'configs/resnet18_aircraft.json',
 #       'configs/densenet121_aircraft.json'
    ]

    for exp_file in experiment_files:
        try:
            with open(exp_file, 'r') as f:
                config_dict = json.load(f)
            run_single_experiment(config_dict)
        except FileNotFoundError:
            print(f"ERROR: Configuration file not found at '{exp_file}'. Skipping.")
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while running experiment from '{exp_file}': {e}")
