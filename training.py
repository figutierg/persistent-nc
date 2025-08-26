import torch
from tqdm import tqdm
import os
import pickle
import json
from metrics import compute_epoch_persistence_diagram, calculate_nc_metrics


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs a single training epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, criterion, device):
    """
    Performs a single evaluation epoch.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100. * correct / total


def train(model, train_loader, train_loader_eval, train_loader_nc, test_loader, criterion, optimizer, scheduler, device,
          n_epochs, snapshot_prefix, snapshot_dir, pd_sample_size, pd_max_edge_length, save_model_history,
          start_epoch=0):
    """
    The main training loop with "epoch 0" evaluation and conditional model saving strategies.
    """

    run_history = []
    # Define paths for temporary checkpoint files
    history_checkpoint_path = os.path.join(snapshot_dir, f"{snapshot_prefix}_full_history.json.checkpoint")
    model_checkpoint_path = os.path.join(snapshot_dir, "models", f"{snapshot_prefix}_model.checkpoint")
    if start_epoch > 0:


        history_file_to_load = os.path.join(snapshot_dir, f"{snapshot_prefix}_full_history.json")
        if os.path.exists(history_file_to_load + ".checkpoint"):
            history_file_to_load += ".checkpoint"

        if os.path.exists(history_file_to_load):
            print(f"Loading previous run history from: {history_file_to_load}")
            with open(history_file_to_load, 'r') as f:
                run_history = json.load(f)
            # Ensure history is not longer than the start epoch
            run_history = [item for item in run_history if item['epoch'] < start_epoch]
        else:
            print(f"WARNING: History file not found for resumed run. Starting with fresh history.")
    if start_epoch == 0:
        # --- "EPOCH 0" EVALUATION ---
        print("--- Starting Epoch 0 (Initial State Evaluation) ---")

        print("Calculating initial metrics on the training set (for loss/acc)...")
        initial_train_loss, initial_train_acc = evaluate(model, train_loader_eval, criterion, device)
        # --- END OF ADDITION ---

        # Validation metrics are still useful to track
        initial_test_loss, initial_test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch 0 Initial State: Test Loss: {initial_test_loss:.4f}, Test Acc: {initial_test_acc:.2f}%")

        # --- NC and Diagrams are now ONLY on the training set ---
        print("Calculating initial metrics on the training set...")
        initial_nc_metrics = calculate_nc_metrics(model, train_loader_nc, device, model.num_classes)


        # Log the initial state
        epoch_0_data = {
            'epoch': 0, 'train_loss': initial_train_loss, 'train_acc': initial_train_acc,
            'test_loss': initial_test_loss, 'test_acc': initial_test_acc,
            **initial_nc_metrics
        }
        run_history.append(epoch_0_data)
        # 4. Save initial diagrams and model state
        print("Computing initial persistence diagrams...")
        initial_diagrams = compute_epoch_persistence_diagram(model, train_loader, device, pd_sample_size, pd_max_edge_length)
        diagram_dir = os.path.join(snapshot_dir, "diagrams")
        if not os.path.exists(diagram_dir):
            os.makedirs(diagram_dir)
        initial_diagram_path = os.path.join(diagram_dir, f"{snapshot_prefix}_epoch_0_diagrams.pkl")
        with open(initial_diagram_path, "wb") as f:
            pickle.dump(initial_diagrams, f)
        print(f"Saved initial diagrams to {initial_diagram_path}")

        if save_model_history:

            snapshot_model_dir = os.path.join(snapshot_dir, "models")
            if not os.path.exists(snapshot_model_dir):
                os.makedirs(snapshot_model_dir)
            initial_snapshot_path = os.path.join(snapshot_model_dir, f"{snapshot_prefix}_epoch_0.pth")
            torch.save(model.state_dict(), initial_snapshot_path)
            print(f"Saved initial model state to {initial_snapshot_path}")

            print("--- Finished Epoch 0 ---")

    # --- Main Training Loop ---
    for epoch in range(1, n_epochs + 1):
        print(f"--- Starting Epoch {epoch}/{n_epochs} ---")
        current_lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion,
                                       device)  # Still evaluate on test set for scheduler
        scheduler.step(train_loss)

        print(
            f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # --- Calculate metrics ONLY on the training set ---
        nc_metrics = calculate_nc_metrics(model, train_loader_nc, device, model.num_classes)
        diagrams = compute_epoch_persistence_diagram(model, train_loader, device, pd_sample_size,
                                                     pd_max_edge_length)


        # Log all metrics for the current epoch
        epoch_data = {
            'epoch': epoch,
            'train_loss': train_loss, 'train_acc': train_acc,
            'test_loss': test_loss, 'test_acc': test_acc,
            'learning_rate': current_lr,
            **nc_metrics
        }
        run_history.append(epoch_data)

        # Save diagrams (always done)
        diagram_path = os.path.join(diagram_dir, f"{snapshot_prefix}_epoch_{epoch}_diagrams.pkl")
        with open(diagram_path, "wb") as f:
            pickle.dump(diagrams, f)

        # Conditional Model Saving Logic
        if save_model_history:
            # Strategy 1: Save model at every single epoch
            snapshot_path = os.path.join(snapshot_model_dir, f"{snapshot_prefix}_epoch_{epoch}.pth")
            torch.save(model.state_dict(), snapshot_path)
            print(f"Saved full model history checkpoint to {snapshot_path}")
        else:
            # Strategy 2: Save a single replaceable checkpoint every 5 epochs
            if epoch % 5 == 0:
                torch.save(model.state_dict(), model_checkpoint_path)
                print(f"Saved replaceable model checkpoint to {model_checkpoint_path}")

        # Metrics History Checkpointing (always done)
        if epoch % 5 == 0:
            print(f"--- Saving metrics history checkpoint at epoch {epoch} ---")
            with open(history_checkpoint_path, 'w') as f:
                json.dump(run_history, f, indent=4)
            print(f"Checkpoint saved to {history_checkpoint_path}")

        print(f"--- Finished Epoch {epoch} ---")

    # --- Final Save and Cleanup After All Epochs ---
    final_history_path = os.path.join(snapshot_dir, f"{snapshot_prefix}_full_history.json")
    print(f"Saving final, complete run history to {final_history_path}...")
    with open(final_history_path, 'w') as f:
        json.dump(run_history, f, indent=4)
    print("Full run history saved successfully.")

    if os.path.exists(history_checkpoint_path):
        os.remove(history_checkpoint_path)
        print(f"Removed temporary history checkpoint file.")

    # --- MODIFIED SECTION ---
    # If not saving model history, remove the final temporary checkpoint upon successful completion.
    if not save_model_history:
        if os.path.exists(model_checkpoint_path):
            os.remove(model_checkpoint_path)
            print(f"Removed temporary model checkpoint file: {model_checkpoint_path}")
    # --- END OF MODIFICATION ---
