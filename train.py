# train.py

import yaml
import mlflow
import os
import json
import torch
import torch.nn as nn
from datetime import datetime

# Import PyTorch project modules
from src.data_pipeline import build_data_pipeline
from src.model_architecture import (
    build_model,
    compile_model_for_transfer_learning,
    compile_model_for_fine_tuning
)
from src.utils.callbacks import create_callbacks


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.long())  # CrossEntropyLoss expects long targets
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target.long())

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# --- 1. Load configuration and setup MLflow ---
print("Loading configuration...")
with open("config/params.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print("Setting up MLflow...")
mlflow.set_experiment(config['experiment_name'])
run_id = f"{config['project_name']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

with mlflow.start_run(run_name=run_id) as run:
    mlflow.log_params(config)
    print(f"MLflow Run ID: {run.info.run_id}")

    # --- 2. Build data pipeline ---
    print("Building data pipelines...")
    train_loader, val_loader, class_names = build_data_pipeline(config)

    num_classes = len(class_names)
    mlflow.log_param("num_classes", num_classes)

    # Save class names for use during prediction
    class_names_path = os.path.join(config['model_output_dir'], run_id, "class_names.json")
    os.makedirs(os.path.dirname(class_names_path), exist_ok=True)
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f)
    mlflow.log_artifact(class_names_path)
    print(f"Saved class names ({num_classes} classes) to {class_names_path}")

    # --- 3. Build and compile model (Phase 1) ---
    print("Building model...")
    model = build_model(config, num_classes=num_classes)

    model, optimizer, criterion = compile_model_for_transfer_learning(model, config)

    # --- 4. Create callbacks ---
    print("Creating callbacks...")
    callbacks, best_model_path = create_callbacks(config, run_id)

    # --- 5. Training Phase 1 (Head Training) ---
    print("\n" + "=" * 50)
    print("STARTING: Phase 1 - Head Training")
    print("=" * 50 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_val_acc = 0.0
    patience_counter = 0
    reduce_lr_counter = 0

    # Track metrics for ReduceLROnPlateau
    best_val_loss_phase1 = float('inf')

    for epoch in range(config['epochs_head']):
        print(f'Epoch {epoch+1}/{config["epochs_head"]}')

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Log metrics to MLflow
        callbacks['log_metrics'](epoch, train_loss, train_acc, val_loss, val_acc, optimizer.param_groups[0]['lr'])

        # Model checkpoint
        is_best = callbacks['checkpoint'](model, val_acc)
        if is_best:
            best_val_acc = val_acc

        # Early stopping
        if callbacks['early_stopping'](val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        # ReduceLROnPlateau simulation
        if val_loss < best_val_loss_phase1:
            best_val_loss_phase1 = val_loss
            reduce_lr_counter = 0
        else:
            reduce_lr_counter += 1
            if reduce_lr_counter >= callbacks['reduce_lr_patience']:
                # Reduce learning rate
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * callbacks['reduce_lr_factor'], 1e-7)
                    if new_lr != old_lr:
                        param_group['lr'] = new_lr
                        print(f"Reducing learning rate from {old_lr:.2e} to {new_lr:.2e}")
                reduce_lr_counter = 0

    # --- 6. Compile model for Phase 2 (Fine-Tuning) ---
    print("\n" + "=" * 50)
    print("STARTING: Phase 2 - Fine-Tuning")
    print("=" * 50 + "\n")

    # Load best weights from Phase 1
    print(f"Loading best weights from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path))

    model, optimizer, criterion = compile_model_for_fine_tuning(model, config)

    # Total epochs (continue from previous training)
    total_epochs = config['epochs_head'] + config['epochs_fine_tune']

    # Reset best validation accuracy for Phase 2
    best_val_acc_phase2 = 0.0
    patience_counter = 0
    reduce_lr_counter = 0
    best_val_loss_phase2 = float('inf')

    # --- 7. Training Phase 2 (Fine-Tuning) ---
    for epoch in range(config['epochs_fine_tune']):
        current_epoch = epoch + config['epochs_head'] + 1
        print(f'Epoch {current_epoch}/{total_epochs}')

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Log metrics to MLflow
        callbacks['log_metrics'](current_epoch, train_loss, train_acc, val_loss, val_acc, optimizer.param_groups[0]['lr'])

        # Model checkpoint
        is_best = callbacks['checkpoint'](model, val_acc)
        if is_best:
            best_val_acc_phase2 = val_acc

        # Early stopping
        if callbacks['early_stopping'](val_loss):
            print(f"Early stopping triggered at epoch {current_epoch}")
            break

        # ReduceLROnPlateau simulation
        if val_loss < best_val_loss_phase2:
            best_val_loss_phase2 = val_loss
            reduce_lr_counter = 0
        else:
            reduce_lr_counter += 1
            if reduce_lr_counter >= callbacks['reduce_lr_patience']:
                # Reduce learning rate
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * callbacks['reduce_lr_factor'], 1e-7)
                    if new_lr != old_lr:
                        param_group['lr'] = new_lr
                        print(f"Reducing learning rate from {old_lr:.2e} to {new_lr:.2e}")
                reduce_lr_counter = 0

    # --- 8. Final save and completion ---
    final_model_path = os.path.join(config['model_output_dir'], run_id, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    mlflow.log_artifact(final_model_path)

    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Best model saved at: {best_model_path}")
    print(f"Final model saved at: {final_model_path}")
    print(f"Run ID: {run_id}")
    print("=" * 50)