# src/utils/callbacks.py

import torch
import os
import mlflow
from datetime import datetime
import json


class ModelCheckpoint:
    """Saves the best model based on validation accuracy."""

    def __init__(self, filepath, monitor='val_acc', save_best_only=True, verbose=1):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best_score = None

    def __call__(self, model, score):
        if self.best_score is None or score > self.best_score:
            if self.verbose:
                print(f"Validation {self.monitor} improved from {self.best_score} to {score}, saving model...")
            self.best_score = score
            torch.save(model.state_dict(), self.filepath)
            return True
        else:
            if self.verbose:
                print(f"Validation {self.monitor} did not improve from {self.best_score}")
            return False


class EarlyStopping:
    """Stops training when a monitored metric has stopped improving."""

    def __init__(self, monitor='val_loss', patience=10, verbose=1, restore_best_weights=True):
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.wait = 0
        self.stopped_epoch = 0

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.monitor == 'val_loss' and score > self.best_score) or \
             (self.monitor != 'val_loss' and score < self.best_score):
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose:
                    print(f"Early stopping triggered after {self.patience} epochs with no improvement")
                return True
        return False


class ReduceLROnPlateau:
    """Reduces learning rate when a metric has stopped improving."""

    def __init__(self, optimizer, monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-7):
        self.optimizer = optimizer
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.min_lr = min_lr
        self.best_score = None
        self.wait = 0

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.monitor == 'val_loss' and score > self.best_score) or \
             (self.monitor != 'val_loss' and score < self.best_score):
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Reduce learning rate
                for param_group in self.optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    if new_lr != old_lr:
                        param_group['lr'] = new_lr
                        if self.verbose:
                            print(f"Reducing learning rate from {old_lr} to {new_lr}")
                self.wait = 0


def create_callbacks(config, run_id):
    """
    Creates a list of standard callbacks for PyTorch training.
    """

    # 1. ModelCheckpoint: Save the best model
    checkpoint_dir = os.path.join(config['model_output_dir'], run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")

    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_acc',
        save_best_only=True,
        verbose=1
    )

    # 2. EarlyStopping: Stop training if no improvement
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config['early_stopping_patience'],
        verbose=1,
        restore_best_weights=True
    )

    # 3. ReduceLROnPlateau: Reduce learning rate when plateau
    # This will be applied separately since it needs optimizer access
    reduce_lr_patience = config['reduce_lr_patience']
    reduce_lr_factor = config['reduce_lr_factor']

    # 4. MLflow logging function
    def log_metrics_to_mlflow(epoch, train_loss, train_acc, val_loss, val_acc, learning_rate=None):
        mlflow.log_metric("epoch", epoch)
        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("val_loss", val_loss)
        mlflow.log_metric("val_accuracy", val_acc)
        if learning_rate is not None:
            mlflow.log_metric("learning_rate", learning_rate)

    return {
        'checkpoint': model_checkpoint,
        'early_stopping': early_stopping,
        'reduce_lr_patience': reduce_lr_patience,
        'reduce_lr_factor': reduce_lr_factor,
        'log_metrics': log_metrics_to_mlflow
    }, checkpoint_path