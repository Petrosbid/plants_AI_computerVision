# src/utils/callbacks.py

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)
import mlflow  # برای ردیابی
import os
from datetime import datetime


def create_callbacks(config, run_id):
    """
    لیستی از Callbackهای استاندارد برای آموزش می‌سازد.
    """

    # 1. ModelCheckpoint: ذخیره بهترین مدل
    checkpoint_dir = os.path.join(config['model_output_dir'], run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.h5")

    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # 2. EarlyStopping: توقف آموزش در صورت عدم پیشرفت
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config['early_stopping_patience'],
        verbose=1,
        restore_best_weights=True  # بازگرداندن بهترین وزن‌ها در انتها
    )

    # 3. ReduceLROnPlateau: کاهش نرخ یادگیری در صورت سکون
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=config['reduce_lr_factor'],
        patience=config['reduce_lr_patience'],
        verbose=1,
        min_lr=1e-7  # حداقل نرخ یادگیری
    )

    # 4. TensorBoard: برای مشاهده نمودارها
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir)

    # 5. MLflow: ردیابی پارامترها و نتایج
    mlflow_callback = mlflow.keras.MLflowCallback()

    callbacks_list = [
        model_checkpoint,
        early_stopping,
        reduce_lr,
        tensorboard_callback,
        mlflow_callback
    ]

    return callbacks_list, checkpoint_path