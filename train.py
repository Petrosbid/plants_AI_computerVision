# train.py

import yaml
import mlflow
import os
import json
from datetime import datetime

# Import ماژول‌های پروژه
from src.data_pipeline import build_data_pipeline
from src.model_architecture import (
    build_model,
    compile_model_for_transfer_learning,
    compile_model_for_fine_tuning
)
from src.utils.callbacks import create_callbacks

# --- 1. بارگذاری تنظیمات و راه‌اندازی MLflow ---
print("Loading configuration...")
with open("config/params.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print("Setting up MLflow...")
mlflow.set_experiment(config['experiment_name'])
run_id = f"{config['project_name']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

with mlflow.start_run(run_name=run_id) as run:
    mlflow.log_params(config)
    print(f"MLflow Run ID: {run.info.run_id}")

    # --- 2. ساخت خط لوله داده ---
    print("Building data pipelines...")
    train_ds, val_ds, class_names = build_data_pipeline(config)

    num_classes = len(class_names)
    mlflow.log_param("num_classes", num_classes)

    # ذخیره نام کلاس‌ها برای استفاده در زمان پیش‌بینی
    class_names_path = os.path.join(config['model_output_dir'], run_id, "class_names.json")
    os.makedirs(os.path.dirname(class_names_path), exist_ok=True)
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f)
    mlflow.log_artifact(class_names_path)
    print(f"Saved class names ({num_classes} classes) to {class_names_path}")

    # --- 3. ساخت و کامپایل مدل (فاز 1) ---
    print("Building model...")
    model = build_model(config, num_classes=num_classes)

    model = compile_model_for_transfer_learning(model, config)

    # --- 4. ساخت Callback ها ---
    print("Creating callbacks...")
    callbacks_list, best_model_path = create_callbacks(config, run_id)

    # --- 5. آموزش فاز 1 (Head Training) ---
    print("\n" + "=" * 50)
    print("STARTING: Phase 1 - Head Training")
    print("=" * 50 + "\n")

    history_head = model.fit(
        train_ds,
        epochs=config['epochs_head'],
        validation_data=val_ds,
        callbacks=callbacks_list
    )

    # --- 6. کامپایل مدل (فاز 2: Fine-Tuning) ---
    print("\n" + "=" * 50)
    print("STARTING: Phase 2 - Fine-Tuning")
    print("=" * 50 + "\n")

    # بارگذاری بهترین وزن‌های از فاز 1
    print(f"Loading best weights from {best_model_path}...")
    model.load_weights(best_model_path)

    model = compile_model_for_fine_tuning(model, config)

    # اپوک‌های کل (ادامه آموزش)
    total_epochs = config['epochs_head'] + config['epochs_fine_tune']

    # --- 7. آموزش فاز 2 (Fine-Tuning) ---
    history_fine_tune = model.fit(
        train_ds,
        epochs=total_epochs,
        initial_epoch=history_head.epoch[-1] + 1,  # ادامه از اپوک قبلی
        validation_data=val_ds,
        callbacks=callbacks_list  # (Callback ها بهترین مدل را مجددا ذخیره و EarlyStopping را مدیریت می‌کنند)
    )

    # --- 8. ذخیره نهایی و پایان ---
    final_model_path = os.path.join(config['model_output_dir'], run_id, "final_model.h5")
    model.save(final_model_path)
    mlflow.log_artifact(final_model_path)

    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Best model saved at: {best_model_path}")
    print(f"Final model saved at: {final_model_path}")
    print(f"Run ID: {run_id}")
    print("=" * 50)